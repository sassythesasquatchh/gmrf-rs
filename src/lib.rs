//! Core Gaussian Markov Random Field types for the Rust port of `GaussianMarkovRandomFields.jl`.
//!
//! This crate mirrors the Julia `GMRF` constructors, precision abstractions, and solver plumbing.
//! It establishes reusable math aliases, a sparse precision representation, and sampling/variance
//! utilities that will be shared across FEM, observation, and visualization crates.

pub mod constrained;
pub mod gmrf;
pub mod linear;
pub mod observation;
pub mod precision;
pub mod solver;
pub mod spacetime;
pub mod types;
pub mod uncertainty;
pub mod vtk;

pub use constrained::{assemble_kkt_matrix, ConstrainedPrecisionSolver};
pub use gmrf::{
    constrained_dense_covariance, ConstrainedVarianceDecomposition, Gmrf,
    TransformedCovarianceDecomposition, TransformedVarianceDecomposition,
};
pub use linear::{
    kronecker, ComposedOperator, LinearOperator, MatrixOperator, OperatorWithSqrt,
    SparseRowOperator,
};
pub use observation::{
    add_sparse, apply_gaussian_observations, apply_gaussian_observations_with_precision,
    apply_linear_observation_terms, apply_linear_observation_terms_with_stats,
    build_linear_observation_matrix, condition_linear_gaussian_with_factor,
    ht_precision_weighted_h, ht_precision_weighted_observations, ht_weighted_h,
    ht_weighted_observations, linear_observation_update, linear_observation_update_with_stats,
    observation_selector, FactoredLinearGaussianPosterior, LinearObservationConditioningStats,
    LinearObservationNoise, LinearObservationStackBuilder, LinearObservationTerm,
    LinearObservationUpdateStats, StackedObservationSystem,
};
pub use precision::{PrecisionOperator, PrecisionStorage};
pub use solver::{
    DirectBackend, IterativeMethod, IterativeSolveReport, JacobiPreconditioner, PreconditionerKind,
    Solver, SolverAlgorithm, SolverConfig,
};
pub use spacetime::{add_sparse_blocks, BlockTridiagonalPrecision, TimeStackedObservationBuilder};
pub use types::{
    CholeskyOrdering, GmrfError, OriginalIndex, Permutation, PermutedIndex, SparseCholeskyFactor,
    SparseCholeskySymbolic, SparseLuFactor, SparseMatrix, Vector,
};
pub use uncertainty::{
    clip_vector_to_prior, estimate_batched_hutchinson_variances,
    estimate_batched_transformed_hutchinson_decomposition,
    estimate_batched_transformed_hutchinson_with_solve, estimate_constrained_mc_variances,
    estimate_constrained_transformed_variances,
    estimate_factored_transformed_variance_weighted_trace, estimate_factored_transformed_variances,
    estimate_hutchinson_transformed_variance_weighted_trace,
    estimate_hutchinson_transformed_variance_weighted_trace_with_solve,
    estimate_hutchinson_transformed_variances, estimate_hutchinson_variances,
    estimate_hutchinson_weighted_covariance_trace,
    estimate_hutchinson_weighted_transformed_covariance_trace,
    estimate_local_rbmc_transformed_variances, estimate_local_rbmc_variances,
    estimate_monte_carlo_constrained_transformed_variances,
    estimate_monte_carlo_constrained_variances, estimate_monte_carlo_transformed_variances,
    estimate_monte_carlo_variances, estimate_transformed_mc_variances, exact_solve_diag,
    exact_solve_diag_with_progress, exact_solve_transformed_diag,
    exact_solve_transformed_diag_with_progress, exact_transformed_variance_weighted_trace,
    exact_weighted_covariance_trace, exact_weighted_transformed_covariance_trace, probe_batch_seed,
    probe_batch_sizes, selected_inverse_diag, selected_inverse_diag_with_diagnostics,
    selected_inverse_entries, selected_inverse_entries_with_limit,
    selected_inverse_transformed_diag, stabilize_removed_variances, stabilize_variances,
    transformed_hutchinson_variances_batch_with_solve, weighted_average_vectors,
    BatchedTransformedVarianceDecomposition, BatchedVarianceEstimate, BlockId, LatentBlockMode,
    LocalRbDiagnostics, LocalRbVarianceEstimate, ProbeBatchConfig, ProbeDistribution,
    SelectedInverseDiagResult, SelectedInverseDiagnostics, SelectedInverseEntries,
    SelectedInverseStatus, SelectedInverseTransformedResult, SparseSelectedInverse,
    SparseSymmetricPattern, StabilizedVariance, TransformedVarianceMode,
    TransformedVarianceWeightedTraceEstimate, VarianceEstimate, VarianceEstimator, VarianceFloor,
    WeightedTraceEstimate,
};
pub use vtk::{
    write_structured_points, write_structured_points_2d, write_structured_points_2d_file,
    write_structured_points_2d_vtu, write_structured_points_2d_vtu_file,
    write_structured_points_vtu,
};
