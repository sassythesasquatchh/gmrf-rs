//! Solver configuration and caching for precision systems.
//!
//! This module mirrors the Julia workflow of configuring linear solvers and reusing
//! factorizations. Direct factorizations are cached, while iterative solvers rely on
//! lightweight preconditioners to keep matrix-free operators usable.

use crate::linear::{LinearOperator, MatrixOperator};
use crate::types::{GmrfError, SparseCholeskyFactor, SparseMatrix, Vector};
use crate::uncertainty::selected_inverse_diag;
use rand::Rng;
use rand_distr::StandardNormal;

/// Available direct solver backends.
#[derive(Clone, Copy, Debug)]
pub enum DirectBackend {
    /// Sparse Cholesky factorization using CSC storage (faer-sparse).
    SparseCholesky,
}

/// Available iterative solver flavors.
#[derive(Clone, Copy, Debug)]
pub enum IterativeMethod {
    /// Conjugate Gradient for symmetric positive definite precisions.
    ConjugateGradient,
}

/// Solver algorithm selection.
#[derive(Clone, Copy, Debug)]
pub enum SolverAlgorithm {
    Direct(DirectBackend),
    Iterative(IterativeMethod),
}

/// Preconditioner selection when running iterative solvers.
#[derive(Clone, Copy, Debug)]
pub enum PreconditionerKind {
    None,
    Jacobi,
}

/// User-facing solver configuration analogous to Julia's `configure_algorithm`.
#[derive(Clone, Copy, Debug)]
pub struct SolverConfig {
    pub algorithm: SolverAlgorithm,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub preconditioner: PreconditionerKind,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            algorithm: SolverAlgorithm::Direct(DirectBackend::SparseCholesky),
            tolerance: 1e-8,
            max_iterations: 1024,
            preconditioner: PreconditionerKind::Jacobi,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IterativeSolveReport {
    pub solution: Vector,
    pub iterations: usize,
    pub final_residual_norm: f64,
    pub converged: bool,
}

/// Cache for direct factorizations.
#[derive(Default)]
pub struct SolverCache {
    sparse_cholesky: Option<SparseCholeskyFactor>,
    sparse_dimension: Option<usize>,
    sparse_matrix_ptr: Option<*const SparseMatrix>,
}

impl SolverCache {
    fn factorize_sparse(&mut self, precision: &SparseMatrix) -> Result<(), GmrfError> {
        let current_ptr = precision as *const SparseMatrix;
        if let (Some(ptr), Some(_chol), Some(dim)) = (
            self.sparse_matrix_ptr,
            self.sparse_cholesky.as_ref(),
            self.sparse_dimension,
        ) {
            if ptr == current_ptr && dim == precision.nrows() {
                return Ok(());
            }
        }

        let chol = SparseCholeskyFactor::factorize(precision)?;
        self.sparse_cholesky = Some(chol);
        self.sparse_dimension = Some(precision.nrows());
        self.sparse_matrix_ptr = Some(current_ptr);
        Ok(())
    }

    fn sparse_cholesky(
        &mut self,
        precision: &SparseMatrix,
    ) -> Result<&SparseCholeskyFactor, GmrfError> {
        self.factorize_sparse(precision)?;
        Ok(self
            .sparse_cholesky
            .as_ref()
            .expect("sparse factorization populated"))
    }

    fn solve_sparse(
        &mut self,
        precision: &SparseMatrix,
        rhs: &Vector,
    ) -> Result<Vector, GmrfError> {
        let factor = self.sparse_cholesky(precision)?;
        let mut out = rhs.clone();
        factor.solve_in_place(&mut out)?;
        Ok(out)
    }

    fn logdet_precision_sparse(&mut self, precision: &SparseMatrix) -> Result<f64, GmrfError> {
        let factor = self.sparse_cholesky(precision)?;
        factor.logdet_precision()
    }

    fn inverse_diag_sparse(&mut self, precision: &SparseMatrix) -> Result<Vector, GmrfError> {
        let factor = self.sparse_cholesky(precision)?;
        Ok(selected_inverse_diag(factor)?.values)
    }

    fn solve_sparse_cholesky_transpose(
        &mut self,
        precision: &SparseMatrix,
        rhs: &Vector,
    ) -> Result<Vector, GmrfError> {
        let factor = self.sparse_cholesky(precision)?;
        let mut out = rhs.clone();
        factor.solve_l_transpose_in_place(&mut out)?;
        Ok(out)
    }
}

/// A lightweight preconditioner for iterative methods.
pub trait Preconditioner: Send + Sync {
    fn dimension(&self) -> usize;
    fn apply(&self, x: &Vector) -> Result<Vector, GmrfError>;
}

/// Diagonal (Jacobi) preconditioner using the inverse diagonal of a sparse matrix.
pub struct JacobiPreconditioner {
    inv_diag: Vector,
}

impl JacobiPreconditioner {
    pub fn from_matrix(matrix: &SparseMatrix) -> Result<Self, GmrfError> {
        let n = matrix.nrows();
        let mut diag = vec![0.0; n];
        for (row, col, value) in matrix.triplet_iter() {
            if row == col {
                diag[row] += *value;
            }
        }

        let mut inv_diag = Vector::zeros(n);
        for (i, value) in diag.into_iter().enumerate() {
            if value.abs() < f64::EPSILON {
                return Err(GmrfError::NonPositiveDefinite);
            }
            inv_diag[i] = 1.0 / value;
        }
        Ok(Self { inv_diag })
    }
}

impl Preconditioner for JacobiPreconditioner {
    fn dimension(&self) -> usize {
        self.inv_diag.len()
    }

    fn apply(&self, x: &Vector) -> Result<Vector, GmrfError> {
        if x.len() != self.inv_diag.len() {
            return Err(GmrfError::DimensionMismatch(
                "preconditioner dimension mismatch",
            ));
        }
        Ok(self.inv_diag.component_mul(x))
    }
}

/// Solver wrapper that orchestrates direct factorization reuse and iterative fallbacks.
pub struct Solver {
    config: SolverConfig,
    cache: SolverCache,
}

impl Solver {
    /// Create a solver with the provided configuration.
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            cache: SolverCache::default(),
        }
    }

    /// Borrow mutable access to the configuration to tweak solver behavior in-place.
    pub fn config_mut(&mut self) -> &mut SolverConfig {
        &mut self.config
    }

    #[cfg(test)]
    pub(crate) fn has_sparse_cholesky_cache(&self) -> bool {
        self.cache.sparse_cholesky.is_some()
    }

    /// Solve `precision * x = rhs` using either a cached direct factorization or an
    /// iterative algorithm.
    pub fn solve_matrix(
        &mut self,
        precision: &SparseMatrix,
        rhs: &Vector,
    ) -> Result<Vector, GmrfError> {
        self.solve_matrix_with_initial_guess(precision, rhs, None)
            .map(|report| report.solution)
    }

    pub fn solve_matrix_with_initial_guess(
        &mut self,
        precision: &SparseMatrix,
        rhs: &Vector,
        initial_guess: Option<&Vector>,
    ) -> Result<IterativeSolveReport, GmrfError> {
        if rhs.len() != precision.nrows() {
            return Err(GmrfError::DimensionMismatch(
                "right hand side length must match precision dimension",
            ));
        }
        if let Some(initial) = initial_guess {
            if initial.len() != rhs.len() {
                return Err(GmrfError::DimensionMismatch(
                    "initial guess length must match right hand side length",
                ));
            }
        }

        match self.config.algorithm {
            SolverAlgorithm::Direct(DirectBackend::SparseCholesky) => {
                let solution = self.cache.solve_sparse(precision, rhs)?;
                let residual_norm = (rhs - &precision.mul_vec(&solution)).norm();
                Ok(IterativeSolveReport {
                    solution,
                    iterations: 0,
                    final_residual_norm: residual_norm,
                    converged: true,
                })
            }
            SolverAlgorithm::Iterative(method) => {
                let operator = MatrixOperator::new(precision.clone());
                let preconditioner = match self.config.preconditioner {
                    PreconditionerKind::None => None,
                    PreconditionerKind::Jacobi => {
                        Some(Box::new(JacobiPreconditioner::from_matrix(precision)?)
                            as Box<dyn Preconditioner>)
                    }
                };
                self.solve_operator_with(
                    method,
                    &operator,
                    preconditioner.as_deref(),
                    rhs,
                    initial_guess,
                )
            }
        }
    }

    /// Solve a matrix-free precision equation using the configured iterative method.
    pub fn solve_operator(
        &mut self,
        operator: &dyn LinearOperator,
        rhs: &Vector,
    ) -> Result<Vector, GmrfError> {
        self.solve_operator_with_initial_guess(operator, rhs, None)
            .map(|report| report.solution)
    }

    pub fn solve_operator_with_initial_guess(
        &mut self,
        operator: &dyn LinearOperator,
        rhs: &Vector,
        initial_guess: Option<&Vector>,
    ) -> Result<IterativeSolveReport, GmrfError> {
        self.solve_operator_with(
            match self.config.algorithm {
                SolverAlgorithm::Direct(_) => IterativeMethod::ConjugateGradient,
                SolverAlgorithm::Iterative(method) => method,
            },
            operator,
            None,
            rhs,
            initial_guess,
        )
    }

    /// Solve `Lᵀ x = rhs` for the cached sparse Cholesky factor `L` of `precision`.
    ///
    /// This ignores the solver configuration because sampling requires a direct factorization.
    pub(crate) fn solve_cholesky_transpose(
        &mut self,
        precision: &SparseMatrix,
        rhs: &Vector,
    ) -> Result<Vector, GmrfError> {
        self.cache.solve_sparse_cholesky_transpose(precision, rhs)
    }

    fn solve_operator_with(
        &self,
        method: IterativeMethod,
        operator: &dyn LinearOperator,
        preconditioner: Option<&dyn Preconditioner>,
        rhs: &Vector,
        initial_guess: Option<&Vector>,
    ) -> Result<IterativeSolveReport, GmrfError> {
        if rhs.len() != operator.dimension() {
            return Err(GmrfError::DimensionMismatch(
                "right hand side length must match operator dimension",
            ));
        }
        if let Some(initial) = initial_guess {
            if initial.len() != rhs.len() {
                return Err(GmrfError::DimensionMismatch(
                    "initial guess length must match right hand side length",
                ));
            }
        }

        let report = match method {
            IterativeMethod::ConjugateGradient => conjugate_gradient_with_diagnostics(
                operator,
                rhs,
                preconditioner,
                initial_guess,
                self.config.max_iterations,
                self.config.tolerance,
            )?,
        };
        if report.converged {
            Ok(report)
        } else {
            Err(GmrfError::IterativeSolverDidNotConverge {
                iterations: report.iterations,
                residual_norm: report.final_residual_norm,
            })
        }
    }

    /// Approximate marginal variances via randomized probing of `diag(Q^{-1})`.
    pub fn approximate_variances<R: Rng + ?Sized>(
        &mut self,
        precision: &SparseMatrix,
        num_probes: usize,
        rng: &mut R,
    ) -> Result<Vector, GmrfError> {
        if num_probes == 0 {
            return Err(GmrfError::DimensionMismatch(
                "at least one probe is required",
            ));
        }

        let dimension = precision.nrows();
        let mut estimates = Vector::zeros(dimension);
        for _ in 0..num_probes {
            let noise = Vector::from_fn(dimension, |_| rng.sample(StandardNormal));
            let solved = self.solve_matrix(precision, &noise)?;
            estimates += solved.component_mul(&noise);
        }

        Ok(estimates / num_probes as f64)
    }

    /// Compute log(det(Σ)) where Σ = Q^{-1}. Mirrors Julia's `logdet_cov` helper.
    pub fn logdet_covariance(&mut self, precision: &SparseMatrix) -> Result<f64, GmrfError> {
        let logdet_q = self.cache.logdet_precision_sparse(precision)?;
        Ok(-logdet_q)
    }

    /// Compute the diagonal of Q^{-1} using the cached Cholesky factor (selected inversion analogue).
    pub fn selected_inverse_diag(&mut self, precision: &SparseMatrix) -> Result<Vector, GmrfError> {
        self.cache.inverse_diag_sparse(precision)
    }
}

impl Default for Solver {
    /// Prefer direct factorizations when available, matching the default solver policy.
    fn default() -> Self {
        Self::new(SolverConfig::default())
    }
}

fn conjugate_gradient_with_diagnostics(
    operator: &dyn LinearOperator,
    rhs: &Vector,
    preconditioner: Option<&dyn Preconditioner>,
    initial_guess: Option<&Vector>,
    max_iterations: usize,
    tolerance: f64,
) -> Result<IterativeSolveReport, GmrfError> {
    let n = rhs.len();
    let mut x = initial_guess.cloned().unwrap_or_else(|| Vector::zeros(n));
    let mut r = rhs - &operator.apply(&x)?;
    let mut residual_norm = r.norm();
    let residual_target = tolerance * rhs.norm().max(1.0);
    if residual_norm <= residual_target {
        return Ok(IterativeSolveReport {
            solution: x,
            iterations: 0,
            final_residual_norm: residual_norm,
            converged: true,
        });
    }
    let mut z = apply_preconditioner(preconditioner, &r)?;
    let mut p = z.clone();
    let mut rz_old = r.dot(&z);

    if !rz_old.is_finite() || rz_old.abs() <= f64::EPSILON {
        return Ok(IterativeSolveReport {
            solution: x,
            iterations: 0,
            final_residual_norm: residual_norm,
            converged: residual_norm <= residual_target,
        });
    }

    for iteration in 0..max_iterations {
        let ap = operator.apply(&p)?;
        let denom = p.dot(&ap);
        if !denom.is_finite() || denom.abs() <= f64::EPSILON {
            return Ok(IterativeSolveReport {
                solution: x,
                iterations: iteration,
                final_residual_norm: residual_norm,
                converged: residual_norm <= residual_target,
            });
        }
        let alpha = rz_old / denom;
        x += alpha * &p;
        r -= alpha * ap;
        residual_norm = r.norm();
        if residual_norm <= residual_target {
            return Ok(IterativeSolveReport {
                solution: x,
                iterations: iteration + 1,
                final_residual_norm: residual_norm,
                converged: true,
            });
        }
        z = apply_preconditioner(preconditioner, &r)?;
        let rz_new = r.dot(&z);
        if !rz_new.is_finite() || rz_new.abs() <= f64::EPSILON {
            return Ok(IterativeSolveReport {
                solution: x,
                iterations: iteration + 1,
                final_residual_norm: residual_norm,
                converged: residual_norm <= residual_target,
            });
        }
        let beta = rz_new / rz_old;
        p = &z + beta * p;
        rz_old = rz_new;
    }

    Ok(IterativeSolveReport {
        solution: x,
        iterations: max_iterations,
        final_residual_norm: residual_norm,
        converged: residual_norm <= residual_target,
    })
}

fn apply_preconditioner(
    preconditioner: Option<&dyn Preconditioner>,
    r: &Vector,
) -> Result<Vector, GmrfError> {
    match preconditioner {
        Some(p) => p.apply(r),
        None => Ok(r.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::LinearOperator;
    use crate::types::CooMatrix;
    use rand::thread_rng;

    fn identity_precision(size: usize) -> SparseMatrix {
        let mut coo = CooMatrix::new(size, size);
        for i in 0..size {
            coo.push(i, i, 1.0);
        }
        SparseMatrix::from(&coo)
    }

    #[test]
    fn direct_solver_uses_cache() {
        let precision = identity_precision(4);
        let rhs = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let mut solver = Solver::default();
        let first = solver.solve_matrix(&precision, &rhs).unwrap();
        let second = solver.solve_matrix(&precision, &rhs).unwrap();
        assert_eq!(first, second);
        assert!(solver.cache.sparse_cholesky.is_some());
        assert_eq!(solver.cache.sparse_dimension, Some(4));
    }

    #[test]
    fn sparse_direct_solver_reuses_factor() {
        let precision = identity_precision(3);
        let rhs = Vector::from_vec(vec![1.0, -1.0, 0.5]);
        let mut solver = Solver::new(SolverConfig {
            algorithm: SolverAlgorithm::Direct(DirectBackend::SparseCholesky),
            ..Default::default()
        });
        let first = solver.solve_matrix(&precision, &rhs).unwrap();
        let second = solver.solve_matrix(&precision, &rhs).unwrap();
        assert_eq!(first, second);
        assert!(solver.cache.sparse_cholesky.is_some());
    }

    #[test]
    fn conjugate_gradient_converges_on_operator() {
        struct IdentityOp;
        impl LinearOperator for IdentityOp {
            fn dimension(&self) -> usize {
                3
            }

            fn apply(&self, x: &Vector) -> Result<Vector, GmrfError> {
                Ok(x.clone())
            }
        }

        let rhs = Vector::from_vec(vec![1.0, -1.0, 0.5]);
        let mut solver = Solver::new(SolverConfig {
            algorithm: SolverAlgorithm::Iterative(IterativeMethod::ConjugateGradient),
            ..Default::default()
        });
        let solution = solver.solve_operator(&IdentityOp, &rhs).unwrap();
        assert!((solution - rhs).norm() < 1e-10);
    }

    #[test]
    fn conjugate_gradient_with_jacobi_matches_cholesky() {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 4.0);
        coo.push(0, 1, 1.0);
        coo.push(1, 0, 1.0);
        coo.push(1, 1, 3.0);
        coo.push(1, 2, 0.5);
        coo.push(2, 1, 0.5);
        coo.push(2, 2, 2.0);
        let precision = SparseMatrix::from(&coo);
        let rhs = Vector::from_vec(vec![1.0, -2.0, 0.5]);

        let mut direct = Solver::new(SolverConfig {
            algorithm: SolverAlgorithm::Direct(DirectBackend::SparseCholesky),
            ..SolverConfig::default()
        });
        let expected = direct.solve_matrix(&precision, &rhs).unwrap();

        let mut cg = Solver::new(SolverConfig {
            algorithm: SolverAlgorithm::Iterative(IterativeMethod::ConjugateGradient),
            tolerance: 1e-12,
            max_iterations: 64,
            preconditioner: PreconditionerKind::Jacobi,
        });
        let report = cg
            .solve_matrix_with_initial_guess(&precision, &rhs, None)
            .unwrap();
        assert!(report.converged);
        assert!(report.iterations > 0);
        assert!((report.solution - expected).norm() <= 1e-10);
    }

    #[test]
    fn warm_started_conjugate_gradient_uses_fewer_iterations_for_nearby_rhs() {
        let mut coo = CooMatrix::new(2, 2);
        coo.push(0, 0, 2.0);
        coo.push(0, 1, 0.25);
        coo.push(1, 0, 0.25);
        coo.push(1, 1, 1.0);
        let precision = SparseMatrix::from(&coo);
        let previous_rhs = Vector::from_vec(vec![1.0, 2.0]);
        let nearby_rhs = Vector::from_vec(vec![1.02, 1.98]);
        let previous_solution = precision
            .cholesky_sqrt_lower()
            .unwrap()
            .solve(&previous_rhs)
            .unwrap();
        let mut solver = Solver::new(SolverConfig {
            algorithm: SolverAlgorithm::Iterative(IterativeMethod::ConjugateGradient),
            tolerance: 1e-12,
            max_iterations: 64,
            preconditioner: PreconditionerKind::Jacobi,
        });
        let cold = solver
            .solve_matrix_with_initial_guess(&precision, &nearby_rhs, None)
            .unwrap();
        let warm = solver
            .solve_matrix_with_initial_guess(&precision, &nearby_rhs, Some(&previous_solution))
            .unwrap();
        assert!(warm.converged);
        assert!(warm.iterations <= cold.iterations);
        assert!(warm.final_residual_norm <= cold.final_residual_norm.max(1e-12));
    }

    #[test]
    fn conjugate_gradient_reports_nonconvergence() {
        let precision = identity_precision(2);
        let rhs = Vector::from_vec(vec![1.0, 2.0]);
        let mut solver = Solver::new(SolverConfig {
            algorithm: SolverAlgorithm::Iterative(IterativeMethod::ConjugateGradient),
            tolerance: 1e-30,
            max_iterations: 0,
            preconditioner: PreconditionerKind::Jacobi,
        });
        let err = solver
            .solve_matrix_with_initial_guess(&precision, &rhs, None)
            .unwrap_err();
        assert!(matches!(
            err,
            GmrfError::IterativeSolverDidNotConverge { .. }
        ));
    }

    #[test]
    fn jacobi_preconditioner_handles_simple_matrix() {
        let precision = identity_precision(2);
        let preconditioner = JacobiPreconditioner::from_matrix(&precision).unwrap();
        let rhs = Vector::from_vec(vec![2.0, -4.0]);
        let applied = preconditioner.apply(&rhs).unwrap();
        assert_eq!(applied, rhs);
    }

    #[test]
    fn variance_estimator_runs_with_probes() {
        let precision = identity_precision(1);
        let mut solver = Solver::default();
        let mut rng = thread_rng();
        let variances = solver
            .approximate_variances(&precision, 8, &mut rng)
            .unwrap();
        assert_eq!(variances.len(), 1);
        assert!(variances[0].is_finite());
    }

    #[test]
    fn logdet_covariance_matches_identity() {
        let precision = identity_precision(3);
        let mut solver = Solver::default();
        let logdet = solver.logdet_covariance(&precision).unwrap();
        // For identity precision, covariance is identity, logdet = 0
        assert!(logdet.abs() < 1e-12);
    }

    #[test]
    fn selected_inverse_diag_matches_identity() {
        let precision = identity_precision(4);
        let mut solver = Solver::default();
        let diag = solver.selected_inverse_diag(&precision).unwrap();
        assert_eq!(diag.len(), 4);
        for v in diag.iter() {
            assert!((*v - 1.0).abs() < 1e-12);
        }
    }
}
