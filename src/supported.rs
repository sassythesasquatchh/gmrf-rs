//! Gaussian probability measures supported on sparse affine subspaces.
//!
//! A [`SupportedGaussianSystem`] represents the energy
//! `0.5 * x^T H x - eta^T x` restricted to `C x = d`.  The ambient energy
//! `H` may be semidefinite.  The probability law is proper when `C` has full
//! row rank and `H` is positive definite on `ker(C)`.
//!
//! Construction certifies these properties without perturbation: sparse
//! Cholesky factorizations of `C C^T` and `H + C^T C` must both succeed.  The
//! latter test is sufficient for arbitrary symmetric `H` and exact for the
//! positive-semidefinite ambient energies used by the FEEC--GMRF graph
//! construction.  Sparse LU of the indefinite KKT matrix is used only for
//! means and covariance actions; the KKT matrix is not a Gaussian precision.

use std::collections::{BTreeMap, BTreeSet};

use crate::linear::SparseRowOperator;
use crate::observation::{add_sparse, multiply_sparse};
use crate::types::{
    CooMatrix, DenseMatrix, GmrfError, SparseCholeskyFactor, SparseLuFactor, SparseMatrix, Vector,
};

const VARIANCE_TOLERANCE: f64 = 1e-10;

/// Ordered, unique coordinates selected from an ambient supported state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedCoordinates {
    ambient_dim: usize,
    indices: Vec<usize>,
}

impl SelectedCoordinates {
    /// Validate an ordered coordinate selection.
    pub fn new(ambient_dim: usize, indices: Vec<usize>) -> Result<Self, GmrfError> {
        let mut seen = BTreeSet::new();
        for index in &indices {
            if *index >= ambient_dim {
                return Err(GmrfError::DimensionMismatch(
                    "selected coordinate exceeds ambient dimension",
                ));
            }
            if !seen.insert(*index) {
                return Err(GmrfError::DimensionMismatch(
                    "selected coordinates must be unique",
                ));
            }
        }
        Ok(Self {
            ambient_dim,
            indices,
        })
    }

    /// Select the leading `selected_dim` ambient coordinates.
    pub fn prefix(ambient_dim: usize, selected_dim: usize) -> Result<Self, GmrfError> {
        if selected_dim > ambient_dim {
            return Err(GmrfError::DimensionMismatch(
                "selected prefix exceeds ambient dimension",
            ));
        }
        Self::new(ambient_dim, (0..selected_dim).collect())
    }

    pub fn ambient_dim(&self) -> usize {
        self.ambient_dim
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Apply the coordinate projection to an ambient vector.
    pub fn project(&self, ambient: &Vector) -> Result<Vector, GmrfError> {
        if ambient.len() != self.ambient_dim {
            return Err(GmrfError::DimensionMismatch(
                "ambient vector length must match coordinate selection",
            ));
        }
        Ok(Vector::from_iterator(
            self.len(),
            self.indices.iter().map(|index| ambient[*index]),
        ))
    }

    fn transpose_apply(&self, selected: &Vector) -> Result<Vector, GmrfError> {
        if selected.len() != self.len() {
            return Err(GmrfError::DimensionMismatch(
                "selected vector length must match coordinate selection",
            ));
        }
        let mut ambient = Vector::zeros(self.ambient_dim);
        for (selected_index, ambient_index) in self.indices.iter().copied().enumerate() {
            ambient[ambient_index] = selected[selected_index];
        }
        Ok(ambient)
    }
}

/// Backward-error diagnostics for the two sparse Cholesky certificates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SupportedGaussianCertificateResiduals {
    /// `None` when there are no constraints, for which full row rank is vacuous.
    pub constraint_gram_relative_residual: Option<f64>,
    pub supported_energy_relative_residual: f64,
}

/// Certificate that the affine support has the requested probability dimension
/// and the quadratic energy is positive on that support.
#[derive(Clone, Debug)]
pub struct SupportedGaussianCertificate {
    constraint_gram_factor: Option<SparseCholeskyFactor>,
    supported_energy_factor: SparseCholeskyFactor,
    residuals: SupportedGaussianCertificateResiduals,
}

impl SupportedGaussianCertificate {
    pub fn constraint_gram_factor_nnz(&self) -> usize {
        self.constraint_gram_factor
            .as_ref()
            .map_or(0, SparseCholeskyFactor::nnz)
    }

    pub fn supported_energy_factor_nnz(&self) -> usize {
        self.supported_energy_factor.nnz()
    }

    pub fn residuals(&self) -> SupportedGaussianCertificateResiduals {
        self.residuals
    }
}

/// Sparse quadratic energy restricted to a sparse affine support.
///
/// This is not an ordinary ambient-space precision: its `energy` may be
/// singular, and the probability dimension is `support_dim()`.
#[derive(Clone, Debug)]
pub struct SupportedGaussianSystem {
    energy: SparseMatrix,
    constraints: SparseMatrix,
    affine_rhs: Vector,
    certificate: SupportedGaussianCertificate,
}

impl SupportedGaussianSystem {
    /// Construct a homogeneous supported Gaussian system `C x = 0`.
    pub fn new(energy: SparseMatrix, constraints: SparseMatrix) -> Result<Self, GmrfError> {
        let affine_rhs = Vector::zeros(constraints.nrows());
        Self::new_affine(energy, constraints, affine_rhs)
    }

    /// Construct an affine supported Gaussian system `C x = d`.
    ///
    /// Full row rank of `C` makes every finite `d` consistent.  Consequently
    /// affine consistency needs no separate least-squares tolerance.
    pub fn new_affine(
        energy: SparseMatrix,
        constraints: SparseMatrix,
        affine_rhs: Vector,
    ) -> Result<Self, GmrfError> {
        validate_dimensions(&energy, &constraints, &affine_rhs)?;
        validate_finite_matrix(&energy, "ambient energy")?;
        validate_finite_matrix(&constraints, "constraint matrix")?;
        validate_finite_vector(&affine_rhs, "affine constraint right-hand side")?;
        validate_exact_symmetry(&energy)?;
        let certificate = certify_supported_energy(&energy, &constraints)?;
        Ok(Self {
            energy,
            constraints,
            affine_rhs,
            certificate,
        })
    }

    pub fn ambient_dim(&self) -> usize {
        self.energy.nrows()
    }

    pub fn constraint_dim(&self) -> usize {
        self.constraints.nrows()
    }

    pub fn support_dim(&self) -> usize {
        self.ambient_dim() - self.constraint_dim()
    }

    pub fn energy(&self) -> &SparseMatrix {
        &self.energy
    }

    pub fn constraints(&self) -> &SparseMatrix {
        &self.constraints
    }

    pub fn affine_rhs(&self) -> &Vector {
        &self.affine_rhs
    }

    pub fn certificate(&self) -> &SupportedGaussianCertificate {
        &self.certificate
    }

    /// Factor the indefinite KKT equations after the probability certificate
    /// has succeeded.
    pub fn factorize(&self) -> Result<SupportedGaussianFactor, GmrfError> {
        SupportedGaussianFactor::factorize(self)
    }
}

/// Sparse solver for means and covariance actions of a certified supported law.
#[derive(Clone, Debug)]
pub struct SupportedGaussianFactor {
    factor: SparseLuFactor,
    constraints: SparseMatrix,
    affine_rhs: Vector,
    ambient_dim: usize,
    support_dim: usize,
    kkt_nnz: usize,
    certificate: SupportedGaussianCertificate,
}

impl SupportedGaussianFactor {
    pub fn factorize(system: &SupportedGaussianSystem) -> Result<Self, GmrfError> {
        let kkt = assemble_supported_kkt_matrix(system.energy(), system.constraints())?;
        let kkt_nnz = kkt.nnz();
        let factor = SparseLuFactor::factorize(&kkt).map_err(|error| match error {
            GmrfError::SingularMatrix => GmrfError::SupportedKktFactorizationFailed,
            other => other,
        })?;
        Ok(Self {
            factor,
            constraints: system.constraints.clone(),
            affine_rhs: system.affine_rhs.clone(),
            ambient_dim: system.ambient_dim(),
            support_dim: system.support_dim(),
            kkt_nnz,
            certificate: system.certificate.clone(),
        })
    }

    pub fn ambient_dim(&self) -> usize {
        self.ambient_dim
    }

    pub fn constraint_dim(&self) -> usize {
        self.constraints.nrows()
    }

    pub fn support_dim(&self) -> usize {
        self.support_dim
    }

    /// Stored nonzeros of the assembled KKT coefficient matrix, before LU fill.
    pub fn kkt_nnz(&self) -> usize {
        self.kkt_nnz
    }

    /// LU fill is not exposed by the current sparse backend wrapper.
    ///
    /// Returning a typed unsupported result avoids misreporting KKT input
    /// nonzeros as factor nonzeros.
    pub fn factor_nnz(&self) -> Result<usize, GmrfError> {
        Err(GmrfError::UnsupportedQuery(
            "the sparse LU backend does not expose numeric factor nonzeros",
        ))
    }

    pub fn certificate(&self) -> &SupportedGaussianCertificate {
        &self.certificate
    }

    /// Solve for the ambient mean under information vector `eta`.
    pub fn solve_mean(&self, information: &Vector) -> Result<Vector, GmrfError> {
        self.solve_kkt(information, &self.affine_rhs)
    }

    /// Apply the covariance of the supported ambient state.
    pub fn covariance_action(&self, ambient_rhs: &Vector) -> Result<Vector, GmrfError> {
        self.solve_kkt(ambient_rhs, &Vector::zeros(self.constraint_dim()))
    }

    /// Solve for a selected-coordinate mean without materializing an ambient covariance.
    pub fn selected_mean(
        &self,
        selection: &SelectedCoordinates,
        information: &Vector,
    ) -> Result<Vector, GmrfError> {
        self.validate_selection(selection)?;
        selection.project(&self.solve_mean(information)?)
    }

    /// Apply `S Cov(x) S^T` to a vector in selected coordinates.
    pub fn selected_covariance_action(
        &self,
        selection: &SelectedCoordinates,
        selected_rhs: &Vector,
    ) -> Result<Vector, GmrfError> {
        self.validate_selection(selection)?;
        let ambient_rhs = selection.transpose_apply(selected_rhs)?;
        selection.project(&self.covariance_action(&ambient_rhs)?)
    }

    /// Materialize the covariance among a deliberately selected set of coordinates.
    pub fn selected_covariance(
        &self,
        selection: &SelectedCoordinates,
    ) -> Result<DenseMatrix, GmrfError> {
        self.validate_selection(selection)?;
        let mut covariance = DenseMatrix::zeros(selection.len(), selection.len());
        for column in 0..selection.len() {
            let basis = Vector::from_fn(selection.len(), |row| usize::from(row == column) as f64);
            let action = self.selected_covariance_action(selection, &basis)?;
            for row in 0..selection.len() {
                covariance[(row, column)] = action[row];
            }
        }
        Ok(covariance)
    }

    /// Exact variances of sparse linear transforms of the ambient supported state.
    pub fn exact_transformed_variances(
        &self,
        operator: &SparseRowOperator,
    ) -> Result<Vector, GmrfError> {
        if operator.ncols != self.ambient_dim {
            return Err(GmrfError::DimensionMismatch(
                "operator columns must match supported ambient dimension",
            ));
        }
        let mut variances = Vector::zeros(operator.nrows());
        for row_index in 0..operator.nrows() {
            let rhs = operator.row_as_vector(row_index)?;
            let action = self.covariance_action(&rhs)?;
            let value = rhs.dot(&action);
            let tolerance = VARIANCE_TOLERANCE * rhs.norm().max(1.0);
            if value < -tolerance {
                return Err(GmrfError::NumericalInstability(
                    "supported transformed variance is materially negative",
                ));
            }
            variances[row_index] = value.max(0.0);
        }
        Ok(variances)
    }

    /// Return `C x - rhs` for support diagnostics.
    ///
    /// Use the stored [`Self::affine_constraint_residual`] for means and a zero
    /// right-hand side for covariance actions.
    pub fn constraint_residual(
        &self,
        ambient: &Vector,
        constraint_rhs: &Vector,
    ) -> Result<Vector, GmrfError> {
        if ambient.len() != self.ambient_dim {
            return Err(GmrfError::DimensionMismatch(
                "ambient vector length must match supported system dimension",
            ));
        }
        if constraint_rhs.len() != self.constraint_dim() {
            return Err(GmrfError::DimensionMismatch(
                "constraint residual right-hand side must match constraint count",
            ));
        }
        validate_finite_vector(ambient, "constraint residual ambient state")?;
        validate_finite_vector(constraint_rhs, "constraint residual right-hand side")?;
        Ok(&self.constraints.mul_vec(ambient) - constraint_rhs)
    }

    /// Return the residual against the affine support stored by the system.
    pub fn affine_constraint_residual(&self, ambient: &Vector) -> Result<Vector, GmrfError> {
        self.constraint_residual(ambient, &self.affine_rhs)
    }

    /// Sampling is intentionally unavailable until a support-aware square-root
    /// construction is implemented.  A KKT solve against ambient white noise
    /// is not a valid sampler in general.
    pub fn sample<R: rand::Rng + ?Sized>(&self, _rng: &mut R) -> Result<Vector, GmrfError> {
        Err(GmrfError::UnsupportedQuery(
            "supported Gaussian sampling requires a certified support square root",
        ))
    }

    fn solve_kkt(
        &self,
        ambient_rhs: &Vector,
        constraint_rhs: &Vector,
    ) -> Result<Vector, GmrfError> {
        if ambient_rhs.len() != self.ambient_dim {
            return Err(GmrfError::DimensionMismatch(
                "ambient right-hand side length must match supported system dimension",
            ));
        }
        if constraint_rhs.len() != self.constraint_dim() {
            return Err(GmrfError::DimensionMismatch(
                "constraint right-hand side length must match supported constraint count",
            ));
        }
        validate_finite_vector(ambient_rhs, "information or covariance right-hand side")?;
        validate_finite_vector(constraint_rhs, "constraint right-hand side")?;

        let mut rhs = Vector::zeros(self.ambient_dim + self.constraint_dim());
        for index in 0..self.ambient_dim {
            rhs[index] = ambient_rhs[index];
        }
        for index in 0..self.constraint_dim() {
            rhs[self.ambient_dim + index] = constraint_rhs[index];
        }
        self.factor.solve_in_place(&mut rhs)?;
        Ok(Vector::from_iterator(
            self.ambient_dim,
            (0..self.ambient_dim).map(|index| rhs[index]),
        ))
    }

    fn validate_selection(&self, selection: &SelectedCoordinates) -> Result<(), GmrfError> {
        if selection.ambient_dim() != self.ambient_dim {
            return Err(GmrfError::DimensionMismatch(
                "coordinate selection ambient dimension must match supported system",
            ));
        }
        Ok(())
    }
}

/// Assemble `[H C^T; C 0]` from exact sparse inputs.
///
/// No magnitude threshold is applied.  The result is an indefinite KKT
/// coefficient matrix, not a Gaussian precision matrix.
pub fn assemble_supported_kkt_matrix(
    energy: &SparseMatrix,
    constraints: &SparseMatrix,
) -> Result<SparseMatrix, GmrfError> {
    if energy.nrows() != energy.ncols() {
        return Err(GmrfError::DimensionMismatch(
            "supported ambient energy must be square",
        ));
    }
    if constraints.ncols() != energy.nrows() {
        return Err(GmrfError::DimensionMismatch(
            "support constraint columns must match ambient dimension",
        ));
    }
    validate_finite_matrix(energy, "ambient energy")?;
    validate_finite_matrix(constraints, "constraint matrix")?;

    let ambient_dim = energy.nrows();
    let constraint_dim = constraints.nrows();
    let mut coo = CooMatrix::new(ambient_dim + constraint_dim, ambient_dim + constraint_dim);
    for (row, col, value) in energy.triplet_iter() {
        if *value != 0.0 {
            coo.push(row, col, *value);
        }
    }
    for (row, col, value) in constraints.triplet_iter() {
        if *value != 0.0 {
            coo.push(col, ambient_dim + row, *value);
            coo.push(ambient_dim + row, col, *value);
        }
    }
    Ok(SparseMatrix::from(&coo))
}

fn validate_dimensions(
    energy: &SparseMatrix,
    constraints: &SparseMatrix,
    affine_rhs: &Vector,
) -> Result<(), GmrfError> {
    if energy.nrows() == 0 {
        return Err(GmrfError::DimensionMismatch(
            "supported ambient dimension must be positive",
        ));
    }
    if energy.nrows() != energy.ncols() {
        return Err(GmrfError::DimensionMismatch(
            "supported ambient energy must be square",
        ));
    }
    if constraints.ncols() != energy.nrows() {
        return Err(GmrfError::DimensionMismatch(
            "support constraint columns must match ambient dimension",
        ));
    }
    if constraints.nrows() > energy.nrows() {
        return Err(GmrfError::RankDeficientSupportConstraints);
    }
    if affine_rhs.len() != constraints.nrows() {
        return Err(GmrfError::DimensionMismatch(
            "affine constraint right-hand side must match constraint rows",
        ));
    }
    Ok(())
}

fn validate_finite_matrix(matrix: &SparseMatrix, role: &'static str) -> Result<(), GmrfError> {
    if matrix
        .triplet_iter()
        .any(|(_, _, value)| !value.is_finite())
    {
        return Err(GmrfError::NonFiniteSupportedInput(role));
    }
    Ok(())
}

fn validate_finite_vector(vector: &Vector, role: &'static str) -> Result<(), GmrfError> {
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(GmrfError::NonFiniteSupportedInput(role));
    }
    Ok(())
}

fn validate_exact_symmetry(energy: &SparseMatrix) -> Result<(), GmrfError> {
    let entries = energy
        .triplet_iter()
        .filter_map(|(row, col, value)| (*value != 0.0).then_some(((row, col), *value)))
        .collect::<BTreeMap<_, _>>();
    for ((row, col), value) in &entries {
        let transpose = entries.get(&(*col, *row)).copied().unwrap_or(0.0);
        if *value != transpose {
            return Err(GmrfError::NonSymmetricSupportedEnergy);
        }
    }
    Ok(())
}

fn certify_supported_energy(
    energy: &SparseMatrix,
    constraints: &SparseMatrix,
) -> Result<SupportedGaussianCertificate, GmrfError> {
    let transpose = sparse_transpose(constraints);
    let constraint_gram = if constraints.nrows() == 0 {
        None
    } else {
        Some(multiply_sparse(constraints, &transpose)?)
    };
    let constraint_gram_factor = constraint_gram
        .as_ref()
        .map(|gram| {
            SparseCholeskyFactor::factorize(gram)
                .map_err(|_| GmrfError::RankDeficientSupportConstraints)
        })
        .transpose()?;
    let transpose_gram = multiply_sparse(&transpose, constraints)?;
    let supported_energy = add_sparse(energy, &transpose_gram);
    let supported_energy_factor = SparseCholeskyFactor::factorize(&supported_energy)
        .map_err(|_| GmrfError::NonPositiveSupportedEnergy)?;

    let residuals = SupportedGaussianCertificateResiduals {
        constraint_gram_relative_residual: constraint_gram
            .as_ref()
            .zip(constraint_gram_factor.as_ref())
            .map(|(matrix, factor)| certificate_solve_residual(matrix, factor))
            .transpose()?,
        supported_energy_relative_residual: certificate_solve_residual(
            &supported_energy,
            &supported_energy_factor,
        )?,
    };

    Ok(SupportedGaussianCertificate {
        constraint_gram_factor,
        supported_energy_factor,
        residuals,
    })
}

fn sparse_transpose(matrix: &SparseMatrix) -> SparseMatrix {
    let mut coo = CooMatrix::new(matrix.ncols(), matrix.nrows());
    for (row, col, value) in matrix.triplet_iter() {
        if *value != 0.0 {
            coo.push(col, row, *value);
        }
    }
    SparseMatrix::from(&coo)
}

fn certificate_solve_residual(
    matrix: &SparseMatrix,
    factor: &SparseCholeskyFactor,
) -> Result<f64, GmrfError> {
    let reference = Vector::from_fn(matrix.ncols(), |index| 1.0 + index as f64 / 7.0);
    let rhs = matrix.mul_vec(&reference);
    let solved = factor.solve(&rhs)?;
    let residual = &matrix.mul_vec(&solved) - &rhs;
    Ok(residual.norm() / rhs.norm().max(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    fn sparse_from_dense(rows: &[&[f64]]) -> SparseMatrix {
        let nrows = rows.len();
        let ncols = rows.first().map_or(0, |row| row.len());
        let mut coo = CooMatrix::new(nrows, ncols);
        for (row, values) in rows.iter().enumerate() {
            assert_eq!(values.len(), ncols);
            for (col, value) in values.iter().copied().enumerate() {
                if value != 0.0 {
                    coo.push(row, col, value);
                }
            }
        }
        SparseMatrix::from(&coo)
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.16e}, expected={expected:.16e}"
        );
    }

    #[test]
    fn semidefinite_ambient_energy_matches_dense_nullspace_reference() {
        // ker([1, 1, 0]) = range([[1, 0], [-1, 0], [0, 1]]).
        let energy = sparse_from_dense(&[&[1.0, 0.0, 0.0], &[0.0, 0.0, 0.0], &[0.0, 0.0, 2.0]]);
        let constraints = sparse_from_dense(&[&[1.0, 1.0, 0.0]]);
        let factor = SupportedGaussianSystem::new(energy, constraints)
            .unwrap()
            .factorize()
            .unwrap();

        let covariance = factor
            .selected_covariance(&SelectedCoordinates::prefix(3, 3).unwrap())
            .unwrap();
        let expected = [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.5]];
        for row in 0..3 {
            for col in 0..3 {
                assert_close(covariance[(row, col)], expected[row][col], 1e-11);
            }
        }
        let action = factor
            .covariance_action(&Vector::from_vec(vec![1.0, 0.0, 0.0]))
            .unwrap();
        assert!(
            factor
                .constraint_residual(&action, &Vector::zeros(1))
                .unwrap()
                .norm()
                < 1e-12
        );
        assert_eq!(factor.support_dim(), 2);
        assert!(
            factor
                .certificate()
                .residuals()
                .supported_energy_relative_residual
                < 1e-12
        );
    }

    #[test]
    fn affine_mean_and_selected_covariance_match_eliminated_reference() {
        let energy = sparse_from_dense(&[&[2.0, 0.0, 0.0], &[0.0, 3.0, 0.0], &[0.0, 0.0, 0.0]]);
        let constraints = sparse_from_dense(&[&[0.0, 0.0, 1.0]]);
        let system =
            SupportedGaussianSystem::new_affine(energy, constraints, Vector::from_vec(vec![4.0]))
                .unwrap();
        let factor = system.factorize().unwrap();
        let information = Vector::from_vec(vec![2.0, -3.0, 7.0]);
        let selection = SelectedCoordinates::new(3, vec![1, 0]).unwrap();

        let mean = factor.selected_mean(&selection, &information).unwrap();
        assert_close(mean[0], -1.0, 1e-12);
        assert_close(mean[1], 1.0, 1e-12);
        let ambient_mean = factor.solve_mean(&information).unwrap();
        assert_close(ambient_mean[2], 4.0, 1e-12);
        assert!(
            factor
                .affine_constraint_residual(&ambient_mean)
                .unwrap()
                .norm()
                < 1e-12
        );

        let covariance = factor.selected_covariance(&selection).unwrap();
        assert_close(covariance[(0, 0)], 1.0 / 3.0, 1e-12);
        assert_close(covariance[(1, 1)], 0.5, 1e-12);
        assert_close(covariance[(0, 1)], 0.0, 1e-12);
    }

    #[test]
    fn transformed_variances_match_dense_covariance() {
        let energy = sparse_from_dense(&[&[1.0, 0.0, 0.0], &[0.0, 0.0, 0.0], &[0.0, 0.0, 2.0]]);
        let constraints = sparse_from_dense(&[&[1.0, 1.0, 0.0]]);
        let factor = SupportedGaussianSystem::new(energy, constraints)
            .unwrap()
            .factorize()
            .unwrap();
        let operator =
            SparseRowOperator::new(3, vec![vec![(0, 1.0), (1, 1.0)], vec![(0, 2.0), (2, -1.0)]])
                .unwrap();
        let variances = factor.exact_transformed_variances(&operator).unwrap();
        assert_close(variances[0], 0.0, 1e-12);
        assert_close(variances[1], 4.5, 1e-11);
    }

    #[test]
    fn empty_constraints_reduce_to_an_spd_precision_solve() {
        let energy = sparse_from_dense(&[&[2.0, 0.5], &[0.5, 1.5]]);
        let constraints = SparseMatrix::from(&CooMatrix::new(0, 2));
        let factor = SupportedGaussianSystem::new(energy, constraints)
            .unwrap()
            .factorize()
            .unwrap();
        let action = factor
            .covariance_action(&Vector::from_vec(vec![1.0, 0.0]))
            .unwrap();
        assert_close(action[0], 6.0 / 11.0, 1e-12);
        assert_close(action[1], -2.0 / 11.0, 1e-12);
        assert_eq!(factor.constraint_dim(), 0);
        assert!(factor
            .certificate()
            .residuals()
            .constraint_gram_relative_residual
            .is_none());
    }

    #[test]
    fn rank_deficient_constraints_are_rejected() {
        let energy = sparse_from_dense(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let constraints = sparse_from_dense(&[&[1.0, 0.0], &[1.0, 0.0]]);
        let error = SupportedGaussianSystem::new(energy, constraints).unwrap_err();
        assert!(matches!(error, GmrfError::RankDeficientSupportConstraints));
    }

    #[test]
    fn improper_restricted_energy_is_rejected() {
        let energy = sparse_from_dense(&[&[1.0, 0.0], &[0.0, 0.0]]);
        let constraints = sparse_from_dense(&[&[1.0, 0.0]]);
        let error = SupportedGaussianSystem::new(energy, constraints).unwrap_err();
        assert!(matches!(error, GmrfError::NonPositiveSupportedEnergy));
    }

    #[test]
    fn indefinite_on_support_is_rejected_even_when_kkt_is_nonsingular() {
        let energy = sparse_from_dense(&[&[1.0, 0.0], &[0.0, -1.0]]);
        let constraints = sparse_from_dense(&[&[1.0, 0.0]]);
        let kkt = assemble_supported_kkt_matrix(&energy, &constraints).unwrap();
        assert!(SparseLuFactor::factorize(&kkt).is_ok());
        let error = SupportedGaussianSystem::new(energy, constraints).unwrap_err();
        assert!(matches!(error, GmrfError::NonPositiveSupportedEnergy));
    }

    #[test]
    fn asymmetric_and_nonfinite_inputs_are_typed_errors() {
        let asymmetric = sparse_from_dense(&[&[1.0, 1.0], &[0.0, 1.0]]);
        let empty = SparseMatrix::from(&CooMatrix::new(0, 2));
        assert!(matches!(
            SupportedGaussianSystem::new(asymmetric, empty).unwrap_err(),
            GmrfError::NonSymmetricSupportedEnergy
        ));

        let nonfinite = sparse_from_dense(&[&[f64::NAN]]);
        let empty = SparseMatrix::from(&CooMatrix::new(0, 1));
        assert!(matches!(
            SupportedGaussianSystem::new(nonfinite, empty).unwrap_err(),
            GmrfError::NonFiniteSupportedInput("ambient energy")
        ));

        let energy = sparse_from_dense(&[&[1.0]]);
        let nonfinite_constraint = sparse_from_dense(&[&[f64::INFINITY]]);
        assert!(matches!(
            SupportedGaussianSystem::new(energy.clone(), nonfinite_constraint).unwrap_err(),
            GmrfError::NonFiniteSupportedInput("constraint matrix")
        ));
        let constraint = sparse_from_dense(&[&[1.0]]);
        assert!(matches!(
            SupportedGaussianSystem::new_affine(
                energy,
                constraint,
                Vector::from_vec(vec![f64::NEG_INFINITY]),
            )
            .unwrap_err(),
            GmrfError::NonFiniteSupportedInput("affine constraint right-hand side")
        ));
    }

    #[test]
    fn dimension_mismatches_and_duplicate_selections_are_rejected() {
        let nonsquare = sparse_from_dense(&[&[1.0, 0.0]]);
        let empty = SparseMatrix::from(&CooMatrix::new(0, 2));
        assert!(matches!(
            SupportedGaussianSystem::new(nonsquare, empty).unwrap_err(),
            GmrfError::DimensionMismatch(_)
        ));
        assert!(SelectedCoordinates::new(2, vec![0, 0]).is_err());
        assert!(SelectedCoordinates::new(2, vec![2]).is_err());

        let energy = sparse_from_dense(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let wrong_columns = sparse_from_dense(&[&[1.0]]);
        assert!(matches!(
            SupportedGaussianSystem::new(energy.clone(), wrong_columns).unwrap_err(),
            GmrfError::DimensionMismatch(_)
        ));
        let constraint = sparse_from_dense(&[&[1.0, 0.0]]);
        assert!(matches!(
            SupportedGaussianSystem::new_affine(
                energy.clone(),
                constraint.clone(),
                Vector::zeros(0),
            )
            .unwrap_err(),
            GmrfError::DimensionMismatch(_)
        ));

        let factor = SupportedGaussianSystem::new(energy, constraint)
            .unwrap()
            .factorize()
            .unwrap();
        assert!(factor.solve_mean(&Vector::zeros(1)).is_err());
        let wrong_selection = SelectedCoordinates::prefix(3, 1).unwrap();
        assert!(factor.selected_covariance(&wrong_selection).is_err());
        assert!(factor
            .constraint_residual(&Vector::zeros(2), &Vector::zeros(0))
            .is_err());
        assert!(matches!(
            factor.solve_mean(&Vector::from_vec(vec![f64::NAN, 0.0])),
            Err(GmrfError::NonFiniteSupportedInput(_))
        ));
    }

    #[test]
    fn exact_kkt_assembly_preserves_tiny_nonzero_constraints() {
        let energy = sparse_from_dense(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let constraints = sparse_from_dense(&[&[1e-14, 1.0]]);
        let kkt = assemble_supported_kkt_matrix(&energy, &constraints).unwrap();
        let upper_tiny = kkt
            .triplet_iter()
            .find_map(|(row, col, value)| ((row, col) == (0, 2)).then_some(*value));
        let lower_tiny = kkt
            .triplet_iter()
            .find_map(|(row, col, value)| ((row, col) == (2, 0)).then_some(*value));
        assert_eq!(upper_tiny, Some(1e-14));
        assert_eq!(lower_tiny, Some(1e-14));
    }

    #[test]
    fn sampling_and_factor_fill_are_explicitly_unsupported() {
        let energy = sparse_from_dense(&[&[1.0]]);
        let constraints = SparseMatrix::from(&CooMatrix::new(0, 1));
        let factor = SupportedGaussianSystem::new(energy, constraints)
            .unwrap()
            .factorize()
            .unwrap();
        let mut rng = StdRng::seed_from_u64(4);
        assert!(matches!(
            factor.sample(&mut rng),
            Err(GmrfError::UnsupportedQuery(_))
        ));
        assert!(matches!(
            factor.factor_nnz(),
            Err(GmrfError::UnsupportedQuery(_))
        ));
    }
}
