//! Core Gaussian Markov Random Field container and helpers.
//!
//! This module mirrors the Julia `GMRF` struct: it stores mean information, precision data,
//! and cached factorizations for repeated solves. Sampling uses the precision factor when
//! available, while stochastic variance estimators can reuse the same factorization.

use crate::linear::SparseRowOperator;
use crate::precision::PrecisionStorage;
use crate::solver::{Solver, SolverConfig};
use crate::types::{DenseMatrix, GmrfError, SparseCholeskyFactor, SparseMatrix, Vector};
use crate::uncertainty::selected_inverse_diag;
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::Solve;
use faer::{Accum, Par, Side};
use rand::Rng;
use rand_distr::StandardNormal;

/// Gaussian Markov Random Field with precision representation and solver cache.
pub struct Gmrf {
    mean: Vector,
    precision: PrecisionStorage,
    q_factor: Option<SparseCholeskyFactor>,
    solver: Solver,
}

/// Exact diagonal variance decomposition for a Gaussian conditioned on linear equalities.
#[derive(Debug, Clone)]
pub struct ConstrainedVarianceDecomposition {
    pub unconstrained_diag: Vector,
    pub constrained_diag: Vector,
    pub removed_diag: Vector,
}

/// Exact or approximate variance decomposition for transformed outputs `A x`.
#[derive(Debug, Clone)]
pub struct TransformedVarianceDecomposition {
    pub unconstrained_diag: Vector,
    pub constrained_diag: Vector,
    pub removed_diag: Vector,
}

/// Exact covariance decomposition for transformed outputs under linear constraints.
#[derive(Debug, Clone)]
pub struct TransformedCovarianceDecomposition {
    pub unconstrained: DenseMatrix,
    pub constrained: DenseMatrix,
    pub removed: DenseMatrix,
}

impl Gmrf {
    /// Default target relative error (one-sigma) used for Hutchinson variance estimates.
    pub const DEFAULT_HUTCHINSON_RELATIVE_ERROR: f64 = 0.1;
    /// Default target relative error (one-sigma) used for Monte Carlo variance estimates.
    pub const DEFAULT_MC_RELATIVE_ERROR: f64 = 0.1;
    const CONSTRAINED_VARIANCE_TOLERANCE: f64 = 1e-10;

    /// Construct a GMRF from a mean vector and sparse precision matrix.
    pub fn from_mean_and_precision(
        mean: Vector,
        precision: SparseMatrix,
    ) -> Result<Self, GmrfError> {
        let dimension = precision.nrows();
        if mean.len() != dimension || precision.ncols() != dimension {
            return Err(GmrfError::DimensionMismatch(
                "mean and precision dimensions must match",
            ));
        }

        Ok(Self {
            mean,
            precision: PrecisionStorage::Matrix(precision),
            q_factor: None,
            solver: Solver::default(),
        })
    }

    /// Construct a GMRF from an information vector (η) and sparse precision matrix (Q).
    pub fn from_information_and_precision(
        information: Vector,
        precision: SparseMatrix,
    ) -> Result<Self, GmrfError> {
        let dimension = precision.nrows();
        if information.len() != dimension || precision.ncols() != dimension {
            return Err(GmrfError::DimensionMismatch(
                "information vector and precision dimensions must match",
            ));
        }

        let mut gmrf = Self {
            mean: Vector::zeros(dimension),
            precision: PrecisionStorage::Matrix(precision),
            q_factor: None,
            solver: Solver::default(),
        };

        let mean = match &gmrf.precision {
            PrecisionStorage::Matrix(mat) => gmrf.solver.solve_matrix(mat, &information)?,
            PrecisionStorage::Operator(_) => return Err(GmrfError::MissingPrecisionMatrix),
        };
        gmrf.mean = mean;

        Ok(gmrf)
    }

    /// Construct a GMRF from an information vector (η), precision matrix (Q), and
    /// a precomputed sparse Cholesky factorization.
    ///
    /// This avoids refactorizing `Q` when you already have a factorization available.
    pub fn from_information_and_precision_with_sqrt(
        information: Vector,
        precision: SparseMatrix,
        q_factor: SparseCholeskyFactor,
    ) -> Result<Self, GmrfError> {
        let dimension = precision.nrows();
        if information.len() != dimension || precision.ncols() != dimension {
            return Err(GmrfError::DimensionMismatch(
                "information vector and precision dimensions must match",
            ));
        }
        if q_factor.dimension() != dimension {
            return Err(GmrfError::DimensionMismatch(
                "precision factor dimension must match mean length",
            ));
        }

        let mean = Self::solve_with_precision_sqrt(&q_factor, &information)?;

        Ok(Self {
            mean,
            precision: PrecisionStorage::Matrix(precision),
            q_factor: Some(q_factor),
            solver: Solver::default(),
        })
    }

    /// Construct a GMRF with a matrix-free precision operator.
    pub fn from_operator(
        mean: Vector,
        operator: Box<dyn crate::precision::PrecisionOperator>,
    ) -> Self {
        let dimension = operator.dimension();
        assert_eq!(
            mean.len(),
            dimension,
            "mean length must match operator dimension",
        );
        Self {
            mean,
            precision: PrecisionStorage::Operator(operator),
            q_factor: None,
            solver: Solver::default(),
        }
    }

    /// Provide a precision factorization to enable sampling without refactorizing Q.
    pub fn with_precision_sqrt(mut self, q_factor: SparseCholeskyFactor) -> Self {
        self.q_factor = Some(q_factor);
        self
    }

    /// Provide a precision factorization after construction.
    pub fn set_precision_sqrt(&mut self, q_factor: SparseCholeskyFactor) {
        self.q_factor = Some(q_factor);
    }

    /// Configure the solver to switch between direct and iterative algorithms.
    pub fn with_solver_config(mut self, config: SolverConfig) -> Self {
        self.solver = Solver::new(config);
        self
    }

    /// Dimension of the latent field.
    pub fn dimension(&self) -> usize {
        self.precision.dimension()
    }

    /// Borrow the mean vector.
    pub fn mean_vector(&self) -> &Vector {
        &self.mean
    }

    /// Access the concrete precision matrix when available.
    /// Returns `None` when the precision is provided as a matrix-free operator.
    pub fn precision_matrix(&self) -> Option<&SparseMatrix> {
        match &self.precision {
            PrecisionStorage::Matrix(mat) => Some(mat),
            PrecisionStorage::Operator(_) => None,
        }
    }

    /// Access the cached precision Cholesky factorization when one was supplied.
    pub fn precision_factor(&self) -> Option<&SparseCholeskyFactor> {
        self.q_factor.as_ref()
    }

    /// Mean accessor.
    pub fn mean(&self) -> &Vector {
        &self.mean
    }

    /// Access the underlying precision storage (matrix or operator).
    pub fn precision(&self) -> &PrecisionStorage {
        &self.precision
    }

    /// Generate a sample from `N(mean, Q^{-1})` using a cached Cholesky factor when available.
    pub fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Vector, GmrfError> {
        if let Some(q_factor) = &self.q_factor {
            return self.sample_with_precision_sqrt(q_factor, rng);
        }

        // TODO delete these paths
        match &self.precision {
            PrecisionStorage::Matrix(precision) => {
                let noise = Vector::from_fn(self.dimension(), |_| rng.sample(StandardNormal));
                let draw = self.solver.solve_cholesky_transpose(precision, &noise)?;
                Ok(&self.mean + draw)
            }
            PrecisionStorage::Operator(operator) => {
                let noise = Vector::from_fn(self.dimension(), |_| rng.sample(StandardNormal));
                let draw = self.solver.solve_operator(operator.as_ref(), &noise)?;
                Ok(&self.mean + draw)
            }
        }
    }

    /// Generate a sample subject to linear equality constraints `A x = b`.
    ///
    /// The constraint matrix `A` is expected to be dense and low-rank. Sampling uses the
    /// exact Gaussian conditioning update
    /// `x = x0 + Q^{-1} A^T (A Q^{-1} A^T)^{-1} (b - A x0)` where `x0 ~ N(mean, Q^{-1})`.
    pub fn sample_constrained<R: Rng + ?Sized>(
        &mut self,
        constraint_matrix: &DenseMatrix,
        constraint_rhs: &Vector,
        rng: &mut R,
    ) -> Result<Vector, GmrfError> {
        self.validate_constraints(constraint_matrix, constraint_rhs)?;
        if constraint_matrix.nrows() == 0 {
            return self.sample(rng);
        }

        let unconstrained = self.sample(rng)?;
        self.apply_linear_constraints(&unconstrained, constraint_matrix, constraint_rhs)
    }

    /// Compute the posterior mean after imposing linear equalities `A x = b`.
    pub fn constrained_mean(
        &mut self,
        constraint_matrix: &DenseMatrix,
        constraint_rhs: &Vector,
    ) -> Result<Vector, GmrfError> {
        self.validate_constraints(constraint_matrix, constraint_rhs)?;
        if constraint_matrix.nrows() == 0 {
            return Ok(self.mean.clone());
        }

        let unconstrained = self.mean.clone();
        self.apply_linear_constraints(&unconstrained, constraint_matrix, constraint_rhs)
    }

    /// Compute exact marginal variance decompositions for transformed outputs `A x`.
    pub fn exact_transformed_variance_decomposition(
        &mut self,
        operator: &SparseRowOperator,
        constraint_matrix: &DenseMatrix,
    ) -> Result<TransformedVarianceDecomposition, GmrfError> {
        self.validate_transformed_operator(operator)?;
        self.validate_constraint_matrix(constraint_matrix)?;

        let mut unconstrained_diag = Vector::zeros(operator.nrows());
        for (row_index, row) in operator.rows.iter().enumerate() {
            let rhs = operator.row_as_vector(row_index)?;
            let solved = self.solve_precision_internal(&rhs)?;
            let variance = row
                .iter()
                .map(|(state_index, weight)| *weight * solved[*state_index])
                .sum::<f64>();
            unconstrained_diag[row_index] = Self::clamp_small_negative(
                variance,
                rhs.norm().max(1.0),
                "transformed marginal variance must be nonnegative",
            )?;
        }

        let removed_diag =
            self.transformed_constraint_correction_diag(operator, constraint_matrix)?;
        let constrained_diag = Self::subtract_removed_variance(&unconstrained_diag, &removed_diag)?;

        Ok(TransformedVarianceDecomposition {
            unconstrained_diag,
            constrained_diag,
            removed_diag,
        })
    }

    /// Compute the exact covariance of transformed outputs `A x`.
    pub fn exact_transformed_covariance(
        &mut self,
        operator: &SparseRowOperator,
    ) -> Result<DenseMatrix, GmrfError> {
        self.validate_transformed_operator(operator)?;

        let output_dim = operator.nrows();
        let mut solved_columns = Vec::with_capacity(output_dim);
        for row_index in 0..operator.nrows() {
            let rhs = operator.row_as_vector(row_index)?;
            solved_columns.push(self.solve_precision_internal(&rhs)?);
        }

        Ok(DenseMatrix::from_fn(output_dim, output_dim, |i, j| {
            operator.rows[i]
                .iter()
                .map(|(state_index, weight)| *weight * solved_columns[j][*state_index])
                .sum::<f64>()
        }))
    }

    /// Compute the low-rank covariance correction induced by constraints for `A x`.
    ///
    /// This returns `A Q^-1 C^T (C Q^-1 C^T)^-1 C Q^-1 A^T`, where `C` is the
    /// dense constraint matrix. It depends only on the constraints, not on the
    /// constraint right-hand side.
    pub fn transformed_covariance_correction(
        &mut self,
        operator: &SparseRowOperator,
        constraint_matrix: &DenseMatrix,
    ) -> Result<DenseMatrix, GmrfError> {
        self.validate_transformed_operator(operator)?;
        self.validate_constraint_matrix(constraint_matrix)?;
        if constraint_matrix.nrows() == 0 {
            return Ok(DenseMatrix::zeros(operator.nrows(), operator.nrows()));
        }

        let covariance_times_constraint_t =
            self.covariance_times_constraint_t(constraint_matrix)?;
        let schur = schur_complement(constraint_matrix, &covariance_times_constraint_t);
        let schur_factor = schur
            .llt(Side::Lower)
            .map_err(|_| GmrfError::SingularConstraintSystem)?;

        let output_dim = operator.nrows();
        let constraint_dim = constraint_matrix.nrows();
        let projected_constraints = DenseMatrix::from_fn(output_dim, constraint_dim, |row, col| {
            operator.rows[row]
                .iter()
                .map(|(state_idx, value)| *value * covariance_times_constraint_t[(*state_idx, col)])
                .sum::<f64>()
        });
        let mut solved_projected_t = DenseMatrix::from_fn(constraint_dim, output_dim, |i, j| {
            projected_constraints[(j, i)]
        });
        schur_factor.solve_in_place(solved_projected_t.as_mut());

        let mut correction = DenseMatrix::zeros(output_dim, output_dim);
        matmul(
            &mut correction,
            Accum::Replace,
            projected_constraints.as_ref(),
            solved_projected_t.as_ref(),
            1.0,
            Par::Seq,
        );
        Ok(correction)
    }

    /// Compute unconstrained, removed, and constrained covariance for `A x`.
    pub fn exact_transformed_covariance_decomposition(
        &mut self,
        operator: &SparseRowOperator,
        constraint_matrix: &DenseMatrix,
    ) -> Result<TransformedCovarianceDecomposition, GmrfError> {
        self.validate_transformed_operator(operator)?;
        self.validate_constraint_matrix(constraint_matrix)?;

        let unconstrained = self.exact_transformed_covariance(operator)?;
        let removed = self.transformed_covariance_correction(operator, constraint_matrix)?;
        let constrained = subtract_removed_covariance(&unconstrained, &removed)?;

        Ok(TransformedCovarianceDecomposition {
            unconstrained,
            constrained,
            removed,
        })
    }

    /// Compute the diagonal of the low-rank transformed covariance correction.
    pub fn transformed_variance_correction_diag(
        &mut self,
        operator: &SparseRowOperator,
        constraint_matrix: &DenseMatrix,
    ) -> Result<Vector, GmrfError> {
        self.validate_transformed_operator(operator)?;
        self.validate_constraint_matrix(constraint_matrix)?;
        self.transformed_constraint_correction_diag(operator, constraint_matrix)
    }

    /// Approximate marginal variance decompositions for transformed outputs `A x`
    /// via Hutchinson probing.
    pub fn hutchinson_transformed_variance_decomposition<R: Rng + ?Sized>(
        &mut self,
        operator: &SparseRowOperator,
        constraint_matrix: &DenseMatrix,
        num_samples: usize,
        rng: &mut R,
    ) -> Result<TransformedVarianceDecomposition, GmrfError> {
        self.validate_transformed_operator(operator)?;
        self.validate_constraint_matrix(constraint_matrix)?;
        if num_samples == 0 {
            return Err(GmrfError::DimensionMismatch(
                "at least one Hutchinson probe is required",
            ));
        }

        let output_dim = operator.nrows();
        let mut unconstrained_diag = Vector::zeros(output_dim);
        for _ in 0..num_samples {
            let probe = Vector::from_fn(output_dim, |_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 });
            let rhs = operator.apply_transpose(&probe)?;
            let solved = self.solve_precision_internal(&rhs)?;
            let projected = operator.apply(&solved)?;
            unconstrained_diag += projected.component_mul(&probe);
        }
        unconstrained_diag =
            Self::stabilize_estimated_variance_diag(&(unconstrained_diag / (num_samples as f64)));

        let removed_diag =
            self.transformed_constraint_correction_diag(operator, constraint_matrix)?;
        let constrained_diag = Self::subtract_removed_variance(&unconstrained_diag, &removed_diag)?;

        Ok(TransformedVarianceDecomposition {
            unconstrained_diag,
            constrained_diag,
            removed_diag,
        })
    }

    /// Compute exact marginal variance diagonals for a Gaussian conditioned on `A x = b`.
    ///
    /// The returned covariance decomposition is independent of `b`; only the constraint matrix
    /// affects the posterior covariance. The result satisfies
    /// `diag(Q^{-1}) = constrained_diag + removed_diag` up to numerical tolerance.
    pub fn exact_constrained_variance_decomposition(
        &mut self,
        constraint_matrix: &DenseMatrix,
    ) -> Result<ConstrainedVarianceDecomposition, GmrfError> {
        self.validate_constraint_matrix(constraint_matrix)?;

        let mut unconstrained_diag = self.exact_inverse_diag()?;
        for i in 0..unconstrained_diag.len() {
            unconstrained_diag[i] = Self::clamp_small_negative(
                unconstrained_diag[i],
                unconstrained_diag[i].abs().max(1.0),
                "unconstrained marginal variance must be nonnegative",
            )?;
        }

        if constraint_matrix.nrows() == 0 {
            return Ok(ConstrainedVarianceDecomposition {
                unconstrained_diag: unconstrained_diag.clone(),
                constrained_diag: unconstrained_diag,
                removed_diag: Vector::zeros(self.dimension()),
            });
        }

        let removed_diag = self.constrained_variance_correction_diag(constraint_matrix)?;
        let dim = self.dimension();
        let mut constrained_diag = Vector::zeros(dim);
        for i in 0..dim {
            let scale = unconstrained_diag[i].abs().max(1.0);
            let max_removed = unconstrained_diag[i] + Self::CONSTRAINED_VARIANCE_TOLERANCE * scale;
            if removed_diag[i] > max_removed {
                return Err(GmrfError::NumericalInstability(
                    "removed marginal variance exceeded unconstrained variance",
                ));
            }
            constrained_diag[i] = Self::clamp_small_negative(
                unconstrained_diag[i] - removed_diag[i].min(unconstrained_diag[i]),
                scale,
                "constrained marginal variance must be nonnegative",
            )?;
        }

        Ok(ConstrainedVarianceDecomposition {
            unconstrained_diag,
            constrained_diag,
            removed_diag,
        })
    }

    /// Compute the low-rank marginal variance correction induced by `A x = b`.
    ///
    /// This returns the diagonal of
    /// `Q^{-1} A^T (A Q^{-1} A^T)^{-1} A Q^{-1}` and depends only on the
    /// constraint matrix, not on `b`.
    pub fn constrained_variance_correction_diag(
        &mut self,
        constraint_matrix: &DenseMatrix,
    ) -> Result<Vector, GmrfError> {
        self.validate_constraint_matrix(constraint_matrix)?;
        if constraint_matrix.nrows() == 0 {
            return Ok(Vector::zeros(self.dimension()));
        }

        let covariance_times_constraint_t =
            self.covariance_times_constraint_t(constraint_matrix)?;
        let schur = schur_complement(constraint_matrix, &covariance_times_constraint_t);
        let schur_factor = schur
            .llt(Side::Lower)
            .map_err(|_| GmrfError::SingularConstraintSystem)?;

        let dim = self.dimension();
        let mut removed_diag = Vector::zeros(dim);
        for i in 0..dim {
            let row = dense_row_as_vector(&covariance_times_constraint_t, i);
            let mut solved = row.clone();
            schur_factor.solve_in_place(solved.as_col_mut().as_mat_mut());
            removed_diag[i] = Self::clamp_small_negative(
                row.dot(&solved),
                row.norm().max(1.0),
                "removed marginal variance must be nonnegative",
            )?;
        }

        Ok(removed_diag)
    }

    /// Generate a sample using a precomputed precision factorization.
    ///
    /// This performs a single sparse triangular solve (plus mean addition), so it is the
    /// fast path when a Cholesky factor is available.
    pub fn sample_with_precision_sqrt<R: Rng + ?Sized>(
        &self,
        q_sqrt: &SparseCholeskyFactor,
        rng: &mut R,
    ) -> Result<Vector, GmrfError> {
        let dimension = q_sqrt.dimension();
        if dimension != self.dimension() {
            return Err(GmrfError::DimensionMismatch(
                "precision factor dimension must match mean length",
            ));
        }

        // We solve Lᵀ x = z (with permutation handling) to sample from Q⁻¹.
        let mut rhs = Vector::from_fn(dimension, |_| rng.sample(StandardNormal));
        q_sqrt.solve_l_transpose_in_place(&mut rhs)?;
        Ok(&self.mean + rhs)
    }

    /// Generate a sample using the stored precision factorization, if available.
    ///
    /// Returns an error if no precision factorization has been provided.
    pub fn sample_one_solve<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vector, GmrfError> {
        let q_sqrt = self
            .q_factor
            .as_ref()
            .ok_or(GmrfError::MissingPrecisionSqrt)?;
        self.sample_with_precision_sqrt(q_sqrt, rng)
    }

    fn solve_with_precision_sqrt(
        q_sqrt: &SparseCholeskyFactor,
        rhs: &Vector,
    ) -> Result<Vector, GmrfError> {
        let dimension = q_sqrt.dimension();
        if rhs.len() != dimension {
            return Err(GmrfError::DimensionMismatch(
                "precision factor dimension must match mean length",
            ));
        }

        q_sqrt.solve(rhs)
    }

    /// Solve `Q x = rhs` using cached factorization when possible.
    pub fn solve_precision(&mut self, rhs: &Vector) -> Result<Vector, GmrfError> {
        self.solve_precision_internal(rhs)
    }

    /// Approximate marginal variances via Hutchinson probing of `diag(Q^{-1})`.
    pub fn hutchinson_variances<R: Rng + ?Sized>(
        &mut self,
        num_samples: usize,
        rng: &mut R,
    ) -> Result<Vector, GmrfError> {
        if num_samples == 0 {
            return Err(GmrfError::DimensionMismatch(
                "at least one probe is required",
            ));
        }

        let dim = self.dimension();
        let mut variances = Vector::zeros(dim);
        for _ in 0..num_samples {
            let probe = Vector::from_fn(dim, |_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 });
            let solved = self.solve_precision(&probe)?;
            variances += solved.component_mul(&probe);
        }

        Ok(variances / (num_samples as f64))
    }

    /// Approximate marginal variances using direct Monte Carlo samples (no Rao-Blackwellisation).
    pub fn mc_variances<R: Rng + ?Sized>(
        &mut self,
        num_samples: usize,
        rng: &mut R,
    ) -> Result<Vector, GmrfError> {
        if num_samples == 0 {
            return Err(GmrfError::DimensionMismatch(
                "at least one sample is required",
            ));
        }

        let dim = self.dimension();
        let mut variances = Vector::zeros(dim);
        for _ in 0..num_samples {
            let draw = self.sample(rng)?;
            let centered = &draw - &self.mean;
            variances += centered.component_mul(&centered);
        }

        Ok(variances / (num_samples as f64))
    }

    /// Compute the number of Hutchinson probe samples needed for a target relative error.
    ///
    /// The Hutchinson estimator averages `z_i (Q^{-1} z)_i` with `z ~ N(0, I)`.
    /// Under the conservative variance bound `Var[z_i (Q^{-1} z)_i] <= 2 * sigma_i^4`
    /// for marginal variance `sigma_i^2`, this yields a one-sigma relative standard
    /// error of at worst `sqrt(2 / n)`, hence `n = ceil(2 / eps^2)`.
    pub fn hutchinson_num_samples(target_rel_error: f64) -> Result<usize, GmrfError> {
        Self::num_samples_for_rel_error(target_rel_error)
    }

    /// Compute the number of Monte Carlo samples needed for a target relative error.
    pub fn mc_num_samples(target_rel_error: f64) -> Result<usize, GmrfError> {
        Self::num_samples_for_rel_error(target_rel_error)
    }

    fn num_samples_for_rel_error(target_rel_error: f64) -> Result<usize, GmrfError> {
        if !target_rel_error.is_finite() || target_rel_error <= 0.0 {
            return Err(GmrfError::DimensionMismatch(
                "target error must be finite and positive",
            ));
        }

        let required = (2.0 / (target_rel_error * target_rel_error)).ceil();
        let samples = required as usize;
        Ok(samples.max(1))
    }

    /// Approximate marginal variances with a Hutchinson probe count derived from a target error.
    ///
    /// Uses `DEFAULT_HUTCHINSON_RELATIVE_ERROR` when `target_rel_error` is `None`.
    pub fn hutchinson_variances_with_error<R: Rng + ?Sized>(
        &mut self,
        target_rel_error: Option<f64>,
        rng: &mut R,
    ) -> Result<Vector, GmrfError> {
        let rel_error = target_rel_error.unwrap_or(Self::DEFAULT_HUTCHINSON_RELATIVE_ERROR);
        let num_samples = Self::hutchinson_num_samples(rel_error)?;
        self.hutchinson_variances(num_samples, rng)
    }

    /// Approximate marginal variances with a Monte Carlo sample count derived from a target error.
    ///
    /// Uses `DEFAULT_MC_RELATIVE_ERROR` when `target_rel_error` is `None`.
    pub fn mc_variances_with_error<R: Rng + ?Sized>(
        &mut self,
        target_rel_error: Option<f64>,
        rng: &mut R,
    ) -> Result<Vector, GmrfError> {
        let rel_error = target_rel_error.unwrap_or(Self::DEFAULT_MC_RELATIVE_ERROR);
        let num_samples = Self::mc_num_samples(rel_error)?;
        self.mc_variances(num_samples, rng)
    }

    fn solve_precision_internal(&mut self, rhs: &Vector) -> Result<Vector, GmrfError> {
        if let Some(q_sqrt) = &self.q_factor {
            return Self::solve_with_precision_sqrt(q_sqrt, rhs);
        }

        match &self.precision {
            PrecisionStorage::Matrix(precision) => self.solver.solve_matrix(precision, rhs),
            PrecisionStorage::Operator(operator) => {
                self.solver.solve_operator(operator.as_ref(), rhs)
            }
        }
    }

    fn validate_constraints(
        &self,
        constraint_matrix: &DenseMatrix,
        constraint_rhs: &Vector,
    ) -> Result<(), GmrfError> {
        self.validate_constraint_matrix(constraint_matrix)?;
        if constraint_matrix.nrows() != constraint_rhs.len() {
            return Err(GmrfError::DimensionMismatch(
                "constraint rhs length must match constraint matrix rows",
            ));
        }
        Ok(())
    }

    fn validate_constraint_matrix(&self, constraint_matrix: &DenseMatrix) -> Result<(), GmrfError> {
        if constraint_matrix.ncols() != self.dimension() {
            return Err(GmrfError::DimensionMismatch(
                "constraint matrix columns must match latent dimension",
            ));
        }
        Ok(())
    }

    fn apply_linear_constraints(
        &mut self,
        unconstrained: &Vector,
        constraint_matrix: &DenseMatrix,
        constraint_rhs: &Vector,
    ) -> Result<Vector, GmrfError> {
        let covariance_times_constraint_t =
            self.covariance_times_constraint_t(constraint_matrix)?;

        let predicted_constraints = dense_matvec(constraint_matrix, unconstrained)?;
        let mut lagrange_rhs = constraint_rhs - &predicted_constraints;

        let schur = schur_complement(constraint_matrix, &covariance_times_constraint_t);
        let schur_factor = schur
            .llt(Side::Lower)
            .map_err(|_| GmrfError::SingularConstraintSystem)?;
        schur_factor.solve_in_place(lagrange_rhs.as_col_mut().as_mat_mut());

        let correction = dense_matvec(&covariance_times_constraint_t, &lagrange_rhs)?;
        Ok(unconstrained + correction)
    }

    fn validate_transformed_operator(&self, operator: &SparseRowOperator) -> Result<(), GmrfError> {
        if operator.ncols != self.dimension() {
            return Err(GmrfError::DimensionMismatch(
                "transformed operator column count must match latent dimension",
            ));
        }
        Ok(())
    }

    fn transformed_constraint_correction_diag(
        &mut self,
        operator: &SparseRowOperator,
        constraint_matrix: &DenseMatrix,
    ) -> Result<Vector, GmrfError> {
        if constraint_matrix.nrows() == 0 {
            return Ok(Vector::zeros(operator.nrows()));
        }

        let covariance_times_constraint_t =
            self.covariance_times_constraint_t(constraint_matrix)?;
        let schur = schur_complement(constraint_matrix, &covariance_times_constraint_t);
        let schur_factor = schur
            .llt(Side::Lower)
            .map_err(|_| GmrfError::SingularConstraintSystem)?;

        let mut removed_diag = Vector::zeros(operator.nrows());
        for (row_index, row) in operator.rows.iter().enumerate() {
            let g = Vector::from_fn(constraint_matrix.nrows(), |constraint_idx| {
                row.iter()
                    .map(|(state_idx, value)| {
                        *value * covariance_times_constraint_t[(*state_idx, constraint_idx)]
                    })
                    .sum::<f64>()
            });

            let mut solved = g.clone();
            schur_factor.solve_in_place(solved.as_col_mut().as_mat_mut());
            removed_diag[row_index] = Self::clamp_small_negative(
                g.dot(&solved),
                g.norm().max(1.0),
                "removed transformed marginal variance must be nonnegative",
            )?;
        }

        Ok(removed_diag)
    }

    fn subtract_removed_variance(
        unconstrained_diag: &Vector,
        removed_diag: &Vector,
    ) -> Result<Vector, GmrfError> {
        if unconstrained_diag.len() != removed_diag.len() {
            return Err(GmrfError::DimensionMismatch(
                "variance vectors must have the same length",
            ));
        }

        let mut constrained_diag = Vector::zeros(unconstrained_diag.len());
        for i in 0..unconstrained_diag.len() {
            let scale = unconstrained_diag[i].abs().max(1.0);
            let max_removed = unconstrained_diag[i] + Self::CONSTRAINED_VARIANCE_TOLERANCE * scale;
            if removed_diag[i] > max_removed {
                return Err(GmrfError::NumericalInstability(
                    "removed transformed marginal variance exceeded unconstrained variance",
                ));
            }
            constrained_diag[i] = Self::clamp_small_negative(
                unconstrained_diag[i] - removed_diag[i].min(unconstrained_diag[i]),
                scale,
                "constrained transformed marginal variance must be nonnegative",
            )?;
        }

        Ok(constrained_diag)
    }

    fn stabilize_estimated_variance_diag(diag: &Vector) -> Vector {
        Vector::from_iterator(diag.len(), diag.iter().map(|value| value.max(0.0)))
    }

    fn exact_inverse_diag(&mut self) -> Result<Vector, GmrfError> {
        if let Some(q_factor) = &self.q_factor {
            return Ok(selected_inverse_diag(q_factor)?.values);
        }

        match &self.precision {
            PrecisionStorage::Matrix(precision) => self.solver.selected_inverse_diag(precision),
            PrecisionStorage::Operator(_) => Err(GmrfError::ExactVarianceRequiresPrecisionMatrix),
        }
    }

    fn covariance_times_constraint_t(
        &mut self,
        constraint_matrix: &DenseMatrix,
    ) -> Result<DenseMatrix, GmrfError> {
        let state_dim = self.dimension();
        let constraint_dim = constraint_matrix.nrows();
        let mut columns = Vec::with_capacity(constraint_dim);
        for row in 0..constraint_dim {
            let rhs = dense_row_as_vector(constraint_matrix, row);
            let solved = self.solve_precision_internal(&rhs)?;
            columns.push(solved);
        }

        Ok(DenseMatrix::from_fn(state_dim, constraint_dim, |i, j| {
            columns[j][i]
        }))
    }

    fn clamp_small_negative(
        value: f64,
        scale: f64,
        message: &'static str,
    ) -> Result<f64, GmrfError> {
        let tol = Self::CONSTRAINED_VARIANCE_TOLERANCE * scale.max(1.0);
        if value >= -tol {
            Ok(value.max(0.0))
        } else {
            Err(GmrfError::NumericalInstability(message))
        }
    }
}

fn dense_matvec(matrix: &DenseMatrix, vector: &Vector) -> Result<Vector, GmrfError> {
    if matrix.ncols() != vector.len() {
        return Err(GmrfError::DimensionMismatch(
            "dense matrix columns must match vector length",
        ));
    }

    let mut out = Vector::zeros(matrix.nrows());
    for (j, col) in matrix.as_ref().col_iter().enumerate() {
        let xj = vector[j];
        if xj == 0.0 {
            continue;
        }
        let col = col
            .try_as_col_major()
            .expect("dense matrix is column-major");
        for (i, value) in col.as_slice().iter().enumerate() {
            out[i] += *value * xj;
        }
    }
    Ok(out)
}

fn dense_row_as_vector(matrix: &DenseMatrix, row: usize) -> Vector {
    let mut out = Vector::zeros(matrix.ncols());
    for (j, col) in matrix.as_ref().col_iter().enumerate() {
        let col = col
            .try_as_col_major()
            .expect("dense matrix is column-major");
        out[j] = col.as_slice()[row];
    }
    out
}

/// Condition an explicit dense covariance by linear constraints `C x = rhs`.
///
/// The result is independent of `rhs` and equals
/// `Sigma - Sigma C^T (C Sigma C^T)^-1 C Sigma`. This is the dense low-rank
/// Schur complement kernel used when callers already have an explicit covariance
/// rather than a precision-backed [`Gmrf`].
pub fn constrained_dense_covariance(
    covariance: &DenseMatrix,
    constraint_matrix: &DenseMatrix,
) -> Result<DenseMatrix, GmrfError> {
    if covariance.nrows() != covariance.ncols() {
        return Err(GmrfError::DimensionMismatch(
            "covariance matrix must be square",
        ));
    }
    if constraint_matrix.ncols() != covariance.nrows() {
        return Err(GmrfError::DimensionMismatch(
            "constraint matrix columns must match covariance dimension",
        ));
    }
    if constraint_matrix.nrows() == 0 {
        return Ok(covariance.clone());
    }

    let state_dim = covariance.nrows();
    let constraint_dim = constraint_matrix.nrows();
    let covariance_times_constraint_t =
        DenseMatrix::from_fn(state_dim, constraint_dim, |state, constraint| {
            (0..state_dim)
                .map(|col| covariance[(state, col)] * constraint_matrix[(constraint, col)])
                .sum::<f64>()
        });
    let schur = schur_complement(constraint_matrix, &covariance_times_constraint_t);
    let schur_factor = schur
        .llt(Side::Lower)
        .map_err(|_| GmrfError::SingularConstraintSystem)?;

    let mut solved_projected_t = DenseMatrix::from_fn(constraint_dim, state_dim, |i, j| {
        covariance_times_constraint_t[(j, i)]
    });
    schur_factor.solve_in_place(solved_projected_t.as_mut());

    let mut correction = DenseMatrix::zeros(state_dim, state_dim);
    matmul(
        &mut correction,
        Accum::Replace,
        covariance_times_constraint_t.as_ref(),
        solved_projected_t.as_ref(),
        1.0,
        Par::Seq,
    );

    subtract_removed_covariance(covariance, &correction)
}

fn subtract_removed_covariance(
    unconstrained: &DenseMatrix,
    removed: &DenseMatrix,
) -> Result<DenseMatrix, GmrfError> {
    if unconstrained.nrows() != removed.nrows() || unconstrained.ncols() != removed.ncols() {
        return Err(GmrfError::DimensionMismatch(
            "covariance matrices must have the same shape",
        ));
    }
    if unconstrained.nrows() != unconstrained.ncols() {
        return Err(GmrfError::DimensionMismatch(
            "transformed covariance must be square",
        ));
    }

    let mut constrained = DenseMatrix::zeros(unconstrained.nrows(), unconstrained.ncols());
    for row in 0..unconstrained.nrows() {
        for col in 0..unconstrained.ncols() {
            constrained[(row, col)] = unconstrained[(row, col)] - removed[(row, col)];
        }
        constrained[(row, row)] = Gmrf::clamp_small_negative(
            constrained[(row, row)],
            unconstrained[(row, row)].abs().max(1.0),
            "constrained transformed covariance diagonal must be nonnegative",
        )?;
    }
    Ok(constrained)
}

fn schur_complement(
    constraint_matrix: &DenseMatrix,
    covariance_times_constraint_t: &DenseMatrix,
) -> DenseMatrix {
    let mut schur = DenseMatrix::zeros(constraint_matrix.nrows(), constraint_matrix.nrows());
    matmul(
        &mut schur,
        Accum::Replace,
        constraint_matrix.as_ref(),
        covariance_times_constraint_t.as_ref(),
        1.0,
        Par::Seq,
    );
    schur
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::{MatrixOperator, SparseRowOperator};
    use crate::types::CooMatrix;
    use rand::rngs::StdRng;
    use rand::thread_rng;
    use rand::SeedableRng;

    fn identity_precision(size: usize) -> SparseMatrix {
        let mut coo = CooMatrix::new(size, size);
        for i in 0..size {
            coo.push(i, i, 1.0);
        }
        SparseMatrix::from(&coo)
    }

    fn assert_dense_close(actual: &DenseMatrix, expected: &DenseMatrix) {
        assert_eq!(actual.nrows(), expected.nrows());
        assert_eq!(actual.ncols(), expected.ncols());
        for row in 0..actual.nrows() {
            for col in 0..actual.ncols() {
                assert!(
                    (actual[(row, col)] - expected[(row, col)]).abs() < 1e-12,
                    "mismatch at ({row}, {col}): {} vs {}",
                    actual[(row, col)],
                    expected[(row, col)]
                );
            }
        }
    }

    #[test]
    fn builds_from_information_vector() {
        let precision = identity_precision(3);
        let info = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let gmrf = Gmrf::from_information_and_precision(info.clone(), precision).unwrap();
        assert_eq!(gmrf.dimension(), 3);
        assert_eq!(gmrf.mean(), &info);
    }

    #[test]
    fn information_constructor_primes_cache() {
        let precision = identity_precision(4);
        let info = Vector::from_vec(vec![1.0, -2.0, 3.0, -4.0]);
        let gmrf = Gmrf::from_information_and_precision(info, precision).unwrap();
        assert!(gmrf.solver.has_sparse_cholesky_cache());
    }

    #[test]
    fn information_constructor_with_sqrt_matches_direct() {
        let precision = identity_precision(3);
        let info = Vector::from_vec(vec![1.0, 0.5, -2.0]);
        let q_sqrt = precision.cholesky_sqrt_lower().unwrap();
        let direct = Gmrf::from_information_and_precision(info.clone(), precision.clone()).unwrap();
        let via_sqrt =
            Gmrf::from_information_and_precision_with_sqrt(info, precision, q_sqrt).unwrap();
        let diff = (direct.mean() - via_sqrt.mean()).norm();
        assert!(diff < 1e-12);
    }

    #[test]
    fn sampling_matches_mean_length() {
        let precision = identity_precision(2);
        let mean = Vector::from_vec(vec![0.5, -0.5]);
        let mut gmrf = Gmrf::from_mean_and_precision(mean.clone(), precision).unwrap();
        let mut rng = thread_rng();
        let draw = gmrf.sample(&mut rng).unwrap();
        assert_eq!(draw.len(), 2);
        assert!((draw[0] - mean[0]).abs() < 10.0); // loose check that draw is finite
    }

    #[test]
    fn hutchinson_returns_reasonable_variance() {
        let precision = identity_precision(1);
        let mean = Vector::from_vec(vec![0.0]);
        let mut gmrf = Gmrf::from_mean_and_precision(mean, precision).unwrap();
        let mut rng = thread_rng();
        let variances = gmrf.hutchinson_variances(32, &mut rng).unwrap();
        assert_eq!(variances.len(), 1);
        assert!(variances[0] > 0.0);
    }

    #[test]
    fn hutchinson_samples_for_target_error() {
        let samples = Gmrf::hutchinson_num_samples(0.1).unwrap();
        assert_eq!(samples, 200);
    }

    #[test]
    fn hutchinson_variances_with_default_error_runs() {
        let precision = identity_precision(1);
        let mean = Vector::from_vec(vec![0.0]);
        let mut gmrf = Gmrf::from_mean_and_precision(mean, precision).unwrap();
        let mut rng = thread_rng();
        let variances = gmrf
            .hutchinson_variances_with_error(None, &mut rng)
            .unwrap();
        assert_eq!(variances.len(), 1);
        assert!(variances[0].is_finite());
    }

    #[test]
    fn hutchinson_rejects_nonpositive_error() {
        let err = Gmrf::hutchinson_num_samples(0.0).unwrap_err();
        assert!(matches!(err, GmrfError::DimensionMismatch(_)));
    }

    #[test]
    fn mc_variances_respects_mean_shift() {
        let precision = identity_precision(1);
        let mean = Vector::from_vec(vec![10.0]);
        let mut gmrf = Gmrf::from_mean_and_precision(mean, precision).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let variances = gmrf.mc_variances(64, &mut rng).unwrap();
        assert!(variances[0] > 0.0);
        assert!(variances[0] < 5.0);
    }

    #[test]
    fn mc_samples_for_target_error() {
        let samples = Gmrf::mc_num_samples(0.1).unwrap();
        assert_eq!(samples, 200);
    }

    #[test]
    fn mc_variances_with_default_error_runs() {
        let precision = identity_precision(1);
        let mean = Vector::from_vec(vec![0.0]);
        let mut gmrf = Gmrf::from_mean_and_precision(mean, precision).unwrap();
        let mut rng = thread_rng();
        let variances = gmrf.mc_variances_with_error(None, &mut rng).unwrap();
        assert_eq!(variances.len(), 1);
        assert!(variances[0].is_finite());
    }

    #[test]
    fn sampling_with_precision_sqrt_reconstructs_noise() {
        let dim = 3;
        let mut coo = CooMatrix::new(dim, dim);
        coo.push(0, 0, 4.0);
        coo.push(1, 1, 9.0);
        coo.push(2, 2, 16.0);
        let precision = SparseMatrix::from(&coo);
        let q_sqrt = precision.cholesky_sqrt_lower().unwrap();

        let mean = Vector::zeros(dim);
        let mut gmrf = Gmrf::from_mean_and_precision(mean, precision)
            .unwrap()
            .with_precision_sqrt(q_sqrt);

        let mut rng = StdRng::seed_from_u64(123);
        let mut rng_expected = rng.clone();
        let noise = Vector::from_fn(dim, |_| rng_expected.sample(StandardNormal));

        let sample = gmrf.sample(&mut rng).unwrap();
        let reconstructed =
            Vector::from_vec(vec![2.0 * sample[0], 3.0 * sample[1], 4.0 * sample[2]]);
        assert!((reconstructed - noise).norm() < 1e-10);
    }

    #[test]
    fn sampling_uses_cholesky_transpose_for_diagonal_precision() {
        let dim = 3;
        let mut coo = CooMatrix::new(dim, dim);
        coo.push(0, 0, 4.0);
        coo.push(1, 1, 9.0);
        coo.push(2, 2, 16.0);
        let precision = SparseMatrix::from(&coo);

        let mean = Vector::zeros(dim);
        let mut gmrf = Gmrf::from_mean_and_precision(mean, precision).unwrap();

        let mut rng = StdRng::seed_from_u64(123);
        let mut rng_expected = rng.clone();
        let noise = Vector::from_fn(dim, |_| rng_expected.sample(StandardNormal));

        let sample = gmrf.sample(&mut rng).unwrap();

        let reconstructed =
            Vector::from_vec(vec![2.0 * sample[0], 3.0 * sample[1], 4.0 * sample[2]]);
        assert!((reconstructed - noise).norm() < 1e-10);
    }

    #[test]
    fn sample_one_solve_requires_sqrt() {
        let precision = identity_precision(2);
        let mean = Vector::zeros(2);
        let gmrf = Gmrf::from_mean_and_precision(mean, precision).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let err = gmrf.sample_one_solve(&mut rng).unwrap_err();
        assert!(matches!(err, GmrfError::MissingPrecisionSqrt));
    }

    #[test]
    fn exact_constrained_variance_decomposition_matches_identity_formula() {
        let precision = identity_precision(2);
        let constraints = DenseMatrix::from_fn(1, 2, |_, j| if j == 0 { 1.0 } else { 2.0 });
        let mut gmrf = Gmrf::from_mean_and_precision(Vector::zeros(2), precision).unwrap();

        let decomposition = gmrf
            .exact_constrained_variance_decomposition(&constraints)
            .unwrap();

        assert!((decomposition.unconstrained_diag[0] - 1.0).abs() < 1e-12);
        assert!((decomposition.unconstrained_diag[1] - 1.0).abs() < 1e-12);
        assert!((decomposition.removed_diag[0] - 0.2).abs() < 1e-12);
        assert!((decomposition.removed_diag[1] - 0.8).abs() < 1e-12);
        assert!((decomposition.constrained_diag[0] - 0.8).abs() < 1e-12);
        assert!((decomposition.constrained_diag[1] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn exact_constrained_variance_decomposition_is_nonnegative_and_ordered() {
        let dim = 3;
        let mut coo = CooMatrix::new(dim, dim);
        coo.push(0, 0, 4.0);
        coo.push(1, 1, 9.0);
        coo.push(2, 2, 16.0);
        let precision = SparseMatrix::from(&coo);
        let constraints = DenseMatrix::from_fn(2, dim, |i, j| match (i, j) {
            (0, 0) => 1.0,
            (0, 1) => -1.0,
            (0, 2) => 0.5,
            (1, 0) => 0.0,
            (1, 1) => 1.0,
            (1, 2) => 1.0,
            _ => 0.0,
        });
        let mut gmrf = Gmrf::from_mean_and_precision(Vector::zeros(dim), precision).unwrap();

        let decomposition = gmrf
            .exact_constrained_variance_decomposition(&constraints)
            .unwrap();

        for i in 0..dim {
            assert!(decomposition.unconstrained_diag[i] >= 0.0);
            assert!(decomposition.removed_diag[i] >= 0.0);
            assert!(decomposition.constrained_diag[i] >= 0.0);
            assert!(
                decomposition.constrained_diag[i] <= decomposition.unconstrained_diag[i] + 1e-12
            );
            assert!(
                (decomposition.unconstrained_diag[i]
                    - decomposition.constrained_diag[i]
                    - decomposition.removed_diag[i])
                    .abs()
                    < 1e-12
            );
        }
    }

    #[test]
    fn exact_constrained_variance_decomposition_rejects_matrix_free_precision() {
        let operator = MatrixOperator::new(identity_precision(2));
        let mut gmrf = Gmrf::from_operator(Vector::zeros(2), Box::new(operator));
        let constraints = DenseMatrix::from_fn(1, 2, |_, j| if j == 0 { 1.0 } else { 0.0 });

        let err = gmrf
            .exact_constrained_variance_decomposition(&constraints)
            .unwrap_err();
        assert!(matches!(
            err,
            GmrfError::ExactVarianceRequiresPrecisionMatrix
        ));
    }

    #[test]
    fn constrained_variance_correction_diag_matches_manual_reference() {
        let mut coo = CooMatrix::new(2, 2);
        coo.push(0, 0, 2.0);
        coo.push(0, 1, 1.0);
        coo.push(1, 0, 1.0);
        coo.push(1, 1, 2.0);
        let precision = SparseMatrix::from(&coo);
        let constraints = DenseMatrix::from_fn(1, 2, |_, j| if j == 0 { 1.0 } else { 0.0 });
        let mut gmrf = Gmrf::from_mean_and_precision(Vector::zeros(2), precision).unwrap();

        let correction = gmrf
            .constrained_variance_correction_diag(&constraints)
            .unwrap();

        assert!((correction[0] - (2.0 / 3.0)).abs() < 1e-12);
        assert!((correction[1] - (1.0 / 6.0)).abs() < 1e-12);
        assert!(correction[0] >= 0.0);
        assert!(correction[1] >= 0.0);
    }

    #[test]
    fn constrained_variance_correction_diag_rank_zero_returns_zero() {
        let precision = identity_precision(3);
        let constraints = DenseMatrix::zeros(0, 3);
        let mut gmrf = Gmrf::from_mean_and_precision(Vector::zeros(3), precision).unwrap();

        let correction = gmrf
            .constrained_variance_correction_diag(&constraints)
            .unwrap();

        assert_eq!(correction.len(), 3);
        assert!(correction.iter().all(|value| value.abs() < 1e-12));
    }

    #[test]
    fn constrained_mean_matches_identity_formula() {
        let precision = identity_precision(2);
        let mut gmrf =
            Gmrf::from_mean_and_precision(Vector::from_vec(vec![1.0, -1.0]), precision).unwrap();
        let constraints = DenseMatrix::from_fn(1, 2, |_, j| if j == 0 { 1.0 } else { -1.0 });
        let rhs = Vector::from_vec(vec![0.25]);

        let mean = gmrf.constrained_mean(&constraints, &rhs).unwrap();

        assert!((mean[0] - mean[1] - 0.25).abs() < 1e-12);
        assert!((mean[0] - 0.125).abs() < 1e-12);
        assert!((mean[1] + 0.125).abs() < 1e-12);
    }

    #[test]
    fn exact_transformed_variance_decomposition_matches_identity_operator() {
        let precision = identity_precision(2);
        let constraints = DenseMatrix::from_fn(1, 2, |_, j| if j == 0 { 1.0 } else { 2.0 });
        let operator = SparseRowOperator::identity(2);
        let mut gmrf = Gmrf::from_mean_and_precision(Vector::zeros(2), precision).unwrap();

        let transformed = gmrf
            .exact_transformed_variance_decomposition(&operator, &constraints)
            .unwrap();
        let latent = gmrf
            .exact_constrained_variance_decomposition(&constraints)
            .unwrap();

        assert!((transformed.unconstrained_diag - latent.unconstrained_diag).norm() < 1e-12);
        assert!((transformed.constrained_diag - latent.constrained_diag).norm() < 1e-12);
        assert!((transformed.removed_diag - latent.removed_diag).norm() < 1e-12);
    }

    #[test]
    fn exact_transformed_variance_decomposition_matches_manual_linear_form() {
        let precision = identity_precision(2);
        let constraints = DenseMatrix::zeros(0, 2);
        let operator = SparseRowOperator::new(2, vec![vec![(0, 1.0), (1, -1.0)]]).unwrap();
        let mut gmrf = Gmrf::from_mean_and_precision(Vector::zeros(2), precision).unwrap();

        let transformed = gmrf
            .exact_transformed_variance_decomposition(&operator, &constraints)
            .unwrap();

        assert_eq!(transformed.unconstrained_diag.len(), 1);
        assert!((transformed.unconstrained_diag[0] - 2.0).abs() < 1e-12);
        assert!((transformed.constrained_diag[0] - 2.0).abs() < 1e-12);
        assert!(transformed.removed_diag[0].abs() < 1e-12);
    }

    #[test]
    fn transformed_covariance_matches_manual_diagonal_precision() {
        let mut coo = CooMatrix::new(2, 2);
        coo.push(0, 0, 2.0);
        coo.push(1, 1, 4.0);
        let precision = SparseMatrix::from(&coo);
        let operator =
            SparseRowOperator::new(2, vec![vec![(0, 1.0), (1, 2.0)], vec![(1, -1.0)]]).unwrap();
        let mut gmrf = Gmrf::from_mean_and_precision(Vector::zeros(2), precision).unwrap();

        let covariance = gmrf.exact_transformed_covariance(&operator).unwrap();
        let expected = DenseMatrix::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 1.5,
            (0, 1) => -0.5,
            (1, 0) => -0.5,
            (1, 1) => 0.25,
            _ => unreachable!(),
        });

        assert_dense_close(&covariance, &expected);
    }

    #[test]
    fn transformed_covariance_decomposition_matches_dense_low_rank_formula() {
        let precision = identity_precision(2);
        let constraints = DenseMatrix::from_fn(1, 2, |_, j| if j == 0 { 1.0 } else { 2.0 });
        let operator =
            SparseRowOperator::new(2, vec![vec![(0, 1.0), (1, -1.0)], vec![(1, 1.0)]]).unwrap();
        let mut gmrf = Gmrf::from_mean_and_precision(Vector::zeros(2), precision).unwrap();

        let decomposition = gmrf
            .exact_transformed_covariance_decomposition(&operator, &constraints)
            .unwrap();

        let unconstrained = DenseMatrix::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 2.0,
            (0, 1) => -1.0,
            (1, 0) => -1.0,
            (1, 1) => 1.0,
            _ => unreachable!(),
        });
        let removed = DenseMatrix::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 0.2,
            (0, 1) => -0.4,
            (1, 0) => -0.4,
            (1, 1) => 0.8,
            _ => unreachable!(),
        });
        let constrained =
            DenseMatrix::from_fn(2, 2, |i, j| unconstrained[(i, j)] - removed[(i, j)]);

        assert_dense_close(&decomposition.unconstrained, &unconstrained);
        assert_dense_close(&decomposition.removed, &removed);
        assert_dense_close(&decomposition.constrained, &constrained);

        let removed_diag = gmrf
            .transformed_variance_correction_diag(&operator, &constraints)
            .unwrap();
        assert!((removed_diag[0] - decomposition.removed[(0, 0)]).abs() < 1e-12);
        assert!((removed_diag[1] - decomposition.removed[(1, 1)]).abs() < 1e-12);
    }

    #[test]
    fn constrained_dense_covariance_matches_manual_low_rank_formula() {
        let covariance = DenseMatrix::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 2.0,
            (0, 1) => 0.5,
            (1, 0) => 0.5,
            (1, 1) => 1.0,
            _ => unreachable!(),
        });
        let constraints = DenseMatrix::from_fn(1, 2, |_, _| 1.0);

        let constrained = constrained_dense_covariance(&covariance, &constraints).unwrap();

        let expected = DenseMatrix::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 0.4375,
            (0, 1) => -0.4375,
            (1, 0) => -0.4375,
            (1, 1) => 0.4375,
            _ => unreachable!(),
        });
        assert_dense_close(&constrained, &expected);
    }

    #[test]
    fn hutchinson_transformed_variance_decomposition_runs() {
        let precision = identity_precision(2);
        let constraints = DenseMatrix::zeros(0, 2);
        let operator = SparseRowOperator::new(2, vec![vec![(0, 1.0)], vec![(1, 1.0)]]).unwrap();
        let mut gmrf = Gmrf::from_mean_and_precision(Vector::zeros(2), precision).unwrap();
        let mut rng = StdRng::seed_from_u64(11);

        let transformed = gmrf
            .hutchinson_transformed_variance_decomposition(&operator, &constraints, 32, &mut rng)
            .unwrap();

        assert_eq!(transformed.unconstrained_diag.len(), 2);
        assert!(transformed
            .unconstrained_diag
            .iter()
            .all(|value| value.is_finite()));
        assert!(transformed
            .constrained_diag
            .iter()
            .all(|value| value.is_finite()));
        assert!(transformed
            .removed_diag
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn stabilize_estimated_variance_diag_clamps_negative_entries() {
        let stabilized =
            Gmrf::stabilize_estimated_variance_diag(&Vector::from_vec(vec![-1.0, 0.5, -1e-9]));
        assert_eq!(stabilized[0], 0.0);
        assert_eq!(stabilized[1], 0.5);
        assert_eq!(stabilized[2], 0.0);
    }

    #[test]
    fn constrained_sample_satisfies_linear_equalities() {
        let dim = 3;
        let mut coo = CooMatrix::new(dim, dim);
        coo.push(0, 0, 4.0);
        coo.push(1, 1, 9.0);
        coo.push(2, 2, 16.0);
        let precision = SparseMatrix::from(&coo);

        let mean = Vector::from_vec(vec![0.3, -0.1, 0.2]);
        let mut gmrf = Gmrf::from_mean_and_precision(mean, precision).unwrap();

        let constraints = DenseMatrix::from_fn(2, dim, |i, j| match (i, j) {
            (0, 0) => 1.0,
            (0, 1) => -1.0,
            (0, 2) => 0.5,
            (1, 0) => 0.0,
            (1, 1) => 1.0,
            (1, 2) => 1.0,
            _ => 0.0,
        });
        let rhs = Vector::from_vec(vec![0.25, -0.4]);

        let mut rng = StdRng::seed_from_u64(17);
        let sample = gmrf
            .sample_constrained(&constraints, &rhs, &mut rng)
            .unwrap();

        let constrained_values = dense_matvec(&constraints, &sample).unwrap();
        assert!((constrained_values - rhs).norm() < 1e-10);
    }

    #[test]
    fn constrained_sampling_matches_identity_projection_formula() {
        let precision = identity_precision(2);
        let constraints = DenseMatrix::from_fn(1, 2, |_, j| if j == 0 { 1.0 } else { 2.0 });
        let rhs = Vector::from_vec(vec![0.5]);

        let mut unconstrained_gmrf =
            Gmrf::from_mean_and_precision(Vector::zeros(2), precision.clone()).unwrap();
        let mut rng_expected = StdRng::seed_from_u64(1234);
        let unconstrained_draw = unconstrained_gmrf.sample(&mut rng_expected).unwrap();

        let mismatch = rhs[0] - (unconstrained_draw[0] + 2.0 * unconstrained_draw[1]);
        let denom = 1.0_f64.powi(2) + 2.0_f64.powi(2);
        let expected = Vector::from_vec(vec![
            unconstrained_draw[0] + mismatch * (1.0 / denom),
            unconstrained_draw[1] + mismatch * (2.0 / denom),
        ]);

        let mut constrained_gmrf =
            Gmrf::from_mean_and_precision(Vector::zeros(2), precision).unwrap();
        let mut rng = StdRng::seed_from_u64(1234);
        let constrained_draw = constrained_gmrf
            .sample_constrained(&constraints, &rhs, &mut rng)
            .unwrap();

        assert!((constrained_draw - expected).norm() < 1e-10);
    }

    #[test]
    fn constrained_sampling_rejects_invalid_constraint_dimensions() {
        let precision = identity_precision(2);
        let mut gmrf = Gmrf::from_mean_and_precision(Vector::zeros(2), precision).unwrap();
        let constraints = DenseMatrix::zeros(1, 3);
        let rhs = Vector::from_vec(vec![0.0]);
        let mut rng = StdRng::seed_from_u64(9);
        let err = gmrf
            .sample_constrained(&constraints, &rhs, &mut rng)
            .unwrap_err();
        assert!(matches!(err, GmrfError::DimensionMismatch(_)));
    }

    #[test]
    fn constrained_sampling_rejects_rank_deficient_constraints() {
        let precision = identity_precision(2);
        let mut gmrf = Gmrf::from_mean_and_precision(Vector::zeros(2), precision).unwrap();
        let constraints = DenseMatrix::from_fn(2, 2, |_, j| if j == 0 { 1.0 } else { 0.0 });
        let rhs = Vector::from_vec(vec![0.0, 0.0]);

        let mut rng = StdRng::seed_from_u64(11);
        let err = gmrf
            .sample_constrained(&constraints, &rhs, &mut rng)
            .unwrap_err();
        assert!(matches!(err, GmrfError::SingularConstraintSystem));
    }
}
