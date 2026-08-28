//! Solvers for sparse precision systems with dense low-rank linear constraints.
//!
//! The KKT factorization provides constrained means and covariance actions for
//! any model expressed by a sparse precision and linear equality constraints.

use crate::linear::SparseRowOperator;
use crate::types::{CooMatrix, DenseMatrix, GmrfError, SparseLuFactor, SparseMatrix, Vector};

const VARIANCE_TOLERANCE: f64 = 1e-10;

/// Sparse KKT solver for `Q x + C^T λ = rhs`, `C x = constraint_rhs`.
#[derive(Debug, Clone)]
pub struct ConstrainedPrecisionSolver {
    factor: SparseLuFactor,
    latent_dim: usize,
    constraint_dim: usize,
    kkt_nnz: usize,
}

impl ConstrainedPrecisionSolver {
    /// Factorize the KKT system for precision `Q` and dense constraints `C`.
    pub fn new(
        precision: &SparseMatrix,
        constraint_matrix: &DenseMatrix,
    ) -> Result<Self, GmrfError> {
        if precision.nrows() != precision.ncols() {
            return Err(GmrfError::DimensionMismatch(
                "constrained precision must be square",
            ));
        }
        if constraint_matrix.ncols() != precision.nrows() {
            return Err(GmrfError::DimensionMismatch(
                "constraint matrix columns must match precision dimension",
            ));
        }

        let kkt = assemble_kkt_matrix(precision, constraint_matrix)?;
        let kkt_nnz = kkt.nnz();
        let factor = SparseLuFactor::factorize(&kkt).map_err(|err| match err {
            GmrfError::SingularMatrix => GmrfError::SingularConstraintSystem,
            other => other,
        })?;
        Ok(Self {
            factor,
            latent_dim: precision.nrows(),
            constraint_dim: constraint_matrix.nrows(),
            kkt_nnz,
        })
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    pub fn constraint_dim(&self) -> usize {
        self.constraint_dim
    }

    /// Stored nonzeros of the assembled KKT coefficient matrix, before LU fill.
    pub fn kkt_nnz(&self) -> usize {
        self.kkt_nnz
    }

    /// LU fill is not exposed by faer's high-level sparse LU wrapper.
    ///
    /// Returning a typed unsupported result avoids reporting KKT input
    /// nonzeros as factor nonzeros.
    pub fn factor_nnz(&self) -> Result<usize, GmrfError> {
        Err(GmrfError::UnsupportedQuery(
            "the sparse LU backend does not expose numeric factor nonzeros",
        ))
    }

    /// Solve the constrained KKT system and return the latent block.
    pub fn solve(&self, latent_rhs: &Vector, constraint_rhs: &Vector) -> Result<Vector, GmrfError> {
        if latent_rhs.len() != self.latent_dim {
            return Err(GmrfError::DimensionMismatch(
                "latent rhs length must match constrained system dimension",
            ));
        }
        if constraint_rhs.len() != self.constraint_dim {
            return Err(GmrfError::DimensionMismatch(
                "constraint rhs length must match constraint count",
            ));
        }

        let mut rhs = Vector::zeros(self.latent_dim + self.constraint_dim);
        for i in 0..self.latent_dim {
            rhs[i] = latent_rhs[i];
        }
        for i in 0..self.constraint_dim {
            rhs[self.latent_dim + i] = constraint_rhs[i];
        }

        self.factor.solve_in_place(&mut rhs)?;
        Ok(Vector::from_iterator(
            self.latent_dim,
            (0..self.latent_dim).map(|i| rhs[i]),
        ))
    }

    /// Solve for the constrained posterior mean/covariance action with zero
    /// constraint right-hand side.
    pub fn solve_mean(&self, latent_rhs: &Vector) -> Result<Vector, GmrfError> {
        self.solve(latent_rhs, &Vector::zeros(self.constraint_dim))
    }

    /// Apply the constrained covariance to a latent right-hand side.
    pub fn solve_covariance_action(&self, latent_rhs: &Vector) -> Result<Vector, GmrfError> {
        self.solve_mean(latent_rhs)
    }

    /// Exact transformed constrained variances for `A x`.
    pub fn exact_transformed_variances(
        &self,
        operator: &SparseRowOperator,
    ) -> Result<Vector, GmrfError> {
        if operator.ncols != self.latent_dim {
            return Err(GmrfError::DimensionMismatch(
                "operator columns must match constrained latent dimension",
            ));
        }

        let mut variances = Vector::zeros(operator.nrows());
        for (row_index, row) in operator.rows.iter().enumerate() {
            let rhs = operator.row_as_vector(row_index)?;
            let solved = self.solve_covariance_action(&rhs)?;
            let value = row
                .iter()
                .map(|(state_index, weight)| *weight * solved[*state_index])
                .sum::<f64>();
            variances[row_index] = clamp_small_negative_variance(
                value,
                rhs.norm().max(1.0),
                "transformed constrained marginal variance must be nonnegative",
            )?;
        }
        Ok(variances)
    }
}

/// Assemble the sparse saddle-point matrix `[Q C^T; C 0]`.
pub fn assemble_kkt_matrix(
    precision: &SparseMatrix,
    constraint_matrix: &DenseMatrix,
) -> Result<SparseMatrix, GmrfError> {
    if precision.nrows() != precision.ncols() {
        return Err(GmrfError::DimensionMismatch(
            "precision matrix must be square",
        ));
    }
    if constraint_matrix.ncols() != precision.nrows() {
        return Err(GmrfError::DimensionMismatch(
            "constraint matrix columns must match precision dimension",
        ));
    }

    let state_dim = precision.nrows();
    let constraint_dim = constraint_matrix.nrows();
    let total_dim = state_dim + constraint_dim;
    let mut coo = CooMatrix::new(total_dim, total_dim);

    for (row, col, value) in precision.triplet_iter() {
        if !value.is_finite() {
            return Err(GmrfError::NumericalInstability(
                "precision matrix contains non-finite entries",
            ));
        }
        if *value != 0.0 {
            coo.push(row, col, *value);
        }
    }

    for i in 0..constraint_dim {
        for j in 0..state_dim {
            let value = constraint_matrix[(i, j)];
            if !value.is_finite() {
                return Err(GmrfError::NumericalInstability(
                    "constraint matrix contains non-finite entries",
                ));
            }
            if value == 0.0 {
                continue;
            }
            coo.push(j, state_dim + i, value);
            coo.push(state_dim + i, j, value);
        }
    }

    Ok(SparseMatrix::from(&coo))
}

fn clamp_small_negative_variance(
    value: f64,
    scale: f64,
    message: &'static str,
) -> Result<f64, GmrfError> {
    let tol = VARIANCE_TOLERANCE * scale.max(1.0);
    if value >= -tol {
        Ok(value.max(0.0))
    } else {
        Err(GmrfError::NumericalInstability(message))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{constrained_dense_covariance, SparseRowOperator};

    fn identity_precision(size: usize) -> SparseMatrix {
        let mut coo = CooMatrix::new(size, size);
        for i in 0..size {
            coo.push(i, i, 1.0);
        }
        SparseMatrix::from(&coo)
    }

    #[test]
    fn constrained_solver_matches_dense_covariance_action() {
        let precision = identity_precision(2);
        let constraints = DenseMatrix::from_fn(1, 2, |_, _| 1.0);
        let solver = ConstrainedPrecisionSolver::new(&precision, &constraints).unwrap();
        assert_eq!(solver.kkt_nnz(), 6);
        assert!(matches!(
            solver.factor_nnz(),
            Err(GmrfError::UnsupportedQuery(_))
        ));
        let rhs = Vector::from_vec(vec![2.0, 0.0]);
        let solved = solver.solve_covariance_action(&rhs).unwrap();

        let dense_covariance = DenseMatrix::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let constrained = constrained_dense_covariance(&dense_covariance, &constraints).unwrap();
        let expected = Vector::from_fn(2, |i| {
            (0..2).map(|j| constrained[(i, j)] * rhs[j]).sum::<f64>()
        });
        assert!((solved - expected).norm() < 1e-10);
    }

    #[test]
    fn exact_transformed_variances_match_dense_formula() {
        let precision = identity_precision(2);
        let constraints = DenseMatrix::from_fn(1, 2, |_, _| 1.0);
        let solver = ConstrainedPrecisionSolver::new(&precision, &constraints).unwrap();
        let operator = SparseRowOperator::identity(2);
        let variances = solver.exact_transformed_variances(&operator).unwrap();
        assert!((variances[0] - 0.5).abs() < 1e-12);
        assert!((variances[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn dense_constraint_kkt_assembly_preserves_tiny_nonzeros() {
        let precision = identity_precision(2);
        let constraints = DenseMatrix::from_fn(1, 2, |_, col| if col == 0 { 1e-14 } else { 1.0 });
        let kkt = assemble_kkt_matrix(&precision, &constraints).unwrap();
        let upper = kkt
            .triplet_iter()
            .find_map(|(row, col, value)| ((row, col) == (0, 2)).then_some(*value));
        let lower = kkt
            .triplet_iter()
            .find_map(|(row, col, value)| ((row, col) == (2, 0)).then_some(*value));
        assert_eq!(upper, Some(1e-14));
        assert_eq!(lower, Some(1e-14));
    }
}
