//! Shared sampling and marginal-variance utilities.
//!
//! This module owns generic Monte Carlo, Hutchinson, local RB, and selected-inverse
//! variance utilities. Downstream FEEC workflows share its batching, seeding,
//! diagnostics, and transformed-operator estimators.

use crate::constrained::ConstrainedPrecisionSolver;
use crate::gmrf::TransformedVarianceDecomposition;
use crate::linear::SparseRowOperator;
use crate::types::{
    DenseMatrix, GmrfError, PermutedIndex, SparseCholeskyFactor, SparseMatrix, Vector,
};
use crate::Gmrf;
use faer::linalg::solvers::Solve;
use faer::Side;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

const BATCH_SEED_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

/// Deterministic stochastic-probe batching configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeBatchConfig {
    pub num_probes: usize,
    pub batch_count: usize,
    pub rng_seed: u64,
}

/// Policy for stabilizing noisy variance estimates.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum VarianceFloor {
    /// Clamp negative roundoff to zero.
    #[default]
    Zero,
    /// Clamp to `max(mean(positive_values), 1) * scale`.
    PositiveMean { scale: f64 },
}

/// A variance vector after applying a floor policy.
#[derive(Debug, Clone, PartialEq)]
pub struct StabilizedVariance {
    pub values: Vector,
    pub floor: f64,
    pub floor_hits: usize,
}

/// Batched scalar/vector variance estimate.
#[derive(Debug, Clone)]
pub struct BatchedVarianceEstimate {
    pub estimate: Vector,
    pub batch_estimates: Vec<Vector>,
    pub batch_sizes: Vec<usize>,
    pub floor_hits: usize,
}

/// Batched scalar estimate for weighted covariance traces.
#[derive(Debug, Clone)]
pub struct WeightedTraceEstimate {
    pub value: f64,
    pub batch_estimates: Vec<f64>,
    pub batch_sizes: Vec<usize>,
    pub batch_standard_error: Option<f64>,
    pub relative_standard_error: Option<f64>,
    pub estimator: VarianceEstimator,
    pub sample_count: usize,
}

/// Paired transformed marginal-variance and weighted-trace estimate.
///
/// The weighted trace is computed from the same transformed variance solves or
/// Hutchinson probe batches, so both diagnostics reuse the same covariance
/// actions.
#[derive(Debug, Clone)]
pub struct TransformedVarianceWeightedTraceEstimate {
    pub variances: VarianceEstimate,
    pub variance_batch_estimates: Vec<Vector>,
    pub weighted_trace: WeightedTraceEstimate,
    pub floor_hits: usize,
}

/// Batched transformed variance decomposition.
#[derive(Debug, Clone)]
pub struct BatchedTransformedVarianceDecomposition {
    pub decomposition: TransformedVarianceDecomposition,
    pub batch_estimates: Vec<Vector>,
    pub batch_sizes: Vec<usize>,
    pub unconstrained_floor_hits: usize,
    pub constrained_floor_hits: usize,
}

/// Variance estimator family used for diagnostics and dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarianceEstimator {
    MonteCarlo,
    ExactSolves,
    LocalRbmc,
    SelectedInverse,
    Hutchinson,
}

impl VarianceEstimator {
    /// Whether this estimator computes exact variances by deterministic solves.
    pub const fn is_exact(self) -> bool {
        matches!(self, Self::ExactSolves)
    }
}

/// Probe distribution for stochastic second-moment estimators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProbeDistribution {
    Gaussian,
    #[default]
    Rademacher,
}

/// Policy for dispatching transformed marginal variance estimators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransformedVarianceMode {
    Exact,
    Hutchinson {
        config: ProbeBatchConfig,
        floor: VarianceFloor,
        distribution: ProbeDistribution,
    },
    Auto {
        exact_max_dofs: usize,
        config: ProbeBatchConfig,
        floor: VarianceFloor,
        distribution: ProbeDistribution,
    },
}

/// Variance estimate plus stochastic diagnostics.
#[derive(Debug, Clone)]
pub struct VarianceEstimate {
    pub values: Vector,
    pub batch_standard_error: Option<Vector>,
    pub relative_standard_error: Option<Vector>,
    pub num_negative: usize,
    pub min_value: f64,
    pub estimator: VarianceEstimator,
    pub sample_count: usize,
    pub batch_sizes: Vec<usize>,
}

/// Extra diagnostics for local Rao-Blackwellised estimators.
#[derive(Debug, Clone)]
pub struct LocalRbDiagnostics {
    pub deterministic_fraction: Vector,
    pub residual_variance_estimate: Vector,
}

/// Local RB estimate with decomposition into deterministic and residual terms.
#[derive(Debug, Clone)]
pub struct LocalRbVarianceEstimate {
    pub estimate: VarianceEstimate,
    pub diagnostics: LocalRbDiagnostics,
}

/// Explicit block identifier for row assignments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub usize);

/// Block specification for local RB estimators.
#[derive(Debug, Clone)]
pub enum LatentBlockMode {
    ContiguousPermuted {
        block_size: usize,
    },
    Explicit {
        blocks: Vec<Vec<PermutedIndex>>,
        row_assignments: Option<Vec<BlockId>>,
    },
}

/// Symmetric sparse pattern stored as lower-triangular `(max, min)` pairs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseSymmetricPattern {
    pub dimension: usize,
    pub pairs: BTreeSet<(usize, usize)>,
}

impl SparseSymmetricPattern {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            pairs: BTreeSet::new(),
        }
    }

    pub fn insert(&mut self, row: usize, col: usize) -> Result<(), GmrfError> {
        if row >= self.dimension || col >= self.dimension {
            return Err(GmrfError::DimensionMismatch(
                "symmetric pattern index exceeds dimension",
            ));
        }
        self.pairs.insert(normalize_pair(row, col));
        Ok(())
    }

    pub fn contains(&self, row: usize, col: usize) -> bool {
        self.pairs.contains(&normalize_pair(row, col))
    }

    pub fn from_cholesky_factor(factor: &SparseCholeskyFactor) -> Self {
        let mut pattern = Self::new(factor.dimension());
        for (row, col, _) in factor.lower_triplets() {
            pattern.pairs.insert(normalize_pair(row, col));
        }
        pattern
    }

    pub fn from_transformed_operator(operator: &SparseRowOperator) -> Result<Self, GmrfError> {
        let mut pattern = Self::new(operator.ncols);
        for row in &operator.rows {
            for (a, _) in row {
                for (b, _) in row {
                    pattern.insert(*a, *b)?;
                }
            }
        }
        Ok(pattern)
    }
}

/// Completion status for a requested selected-inverse computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedInverseStatus {
    Complete,
    ClosureTooLarge,
}

/// Diagnostics for requested selected-inverse entries.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedInverseDiagnostics {
    pub requested_pairs: usize,
    pub closure_pairs: usize,
    pub factor_pairs: usize,
    pub closure_over_factor: f64,
    pub closure_limit: usize,
    pub status: SelectedInverseStatus,
}

impl SelectedInverseDiagnostics {
    pub fn is_complete(&self) -> bool {
        self.status == SelectedInverseStatus::Complete
    }
}

#[derive(Debug, Clone)]
struct SparseSelectedInverseColumn {
    rows: Vec<usize>,
    values: Vec<f64>,
}

/// Sparse selected inverse entries in permuted coordinates.
#[derive(Debug, Clone)]
pub struct SparseSelectedInverse {
    pub dimension: usize,
    columns: Vec<SparseSelectedInverseColumn>,
}

impl SparseSelectedInverse {
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        let (row, col) = normalize_pair(row, col);
        let column = self.columns.get(col)?;
        let index = column.rows.binary_search(&row).ok()?;
        column.values.get(index).copied()
    }

    pub fn diag(&self) -> Vector {
        Vector::from_fn(self.dimension, |i| self.get(i, i).unwrap_or(0.0))
    }
}

/// Selected-inverse result for a requested sparse pattern.
#[derive(Debug, Clone)]
pub struct SelectedInverseEntries {
    pub inverse: Option<SparseSelectedInverse>,
    pub diagnostics: SelectedInverseDiagnostics,
}

/// Selected-inverse latent diagonal result with closure diagnostics.
#[derive(Debug, Clone)]
pub struct SelectedInverseDiagResult {
    pub estimate: VarianceEstimate,
    pub diagnostics: SelectedInverseDiagnostics,
}

/// Selected-inverse transformed variance result. `estimate` is absent when the
/// requested Takahashi closure exceeds the configured entry limit.
#[derive(Debug, Clone)]
pub struct SelectedInverseTransformedResult {
    pub estimate: Option<VarianceEstimate>,
    pub diagnostics: SelectedInverseDiagnostics,
}

/// Split a probe count into deterministic, nearly equal batch sizes.
pub fn probe_batch_sizes(num_probes: usize, batch_count: usize) -> Result<Vec<usize>, GmrfError> {
    if num_probes == 0 {
        return Err(GmrfError::DimensionMismatch(
            "at least one stochastic probe/sample is required",
        ));
    }
    if batch_count == 0 {
        return Err(GmrfError::DimensionMismatch(
            "at least one stochastic batch is required",
        ));
    }

    let batches = batch_count.min(num_probes);
    let base = num_probes / batches;
    let remainder = num_probes % batches;
    Ok((0..batches)
        .map(|index| base + usize::from(index < remainder))
        .collect())
}

/// Seed used for a one-indexed deterministic stochastic batch.
pub fn probe_batch_seed(base_seed: u64, batch_index: usize) -> u64 {
    base_seed.wrapping_add(BATCH_SEED_STRIDE.wrapping_mul((batch_index as u64).wrapping_add(1)))
}

/// Weighted average of aligned variance vectors.
pub fn weighted_average_vectors(
    vectors: &[Vector],
    weights: &[usize],
) -> Result<Vector, GmrfError> {
    if vectors.is_empty() || vectors.len() != weights.len() {
        return Err(GmrfError::DimensionMismatch(
            "variance batches and weights must be non-empty and aligned",
        ));
    }
    let dimension = vectors[0].len();
    if vectors.iter().any(|vector| vector.len() != dimension) {
        return Err(GmrfError::DimensionMismatch(
            "variance batch dimensions must match",
        ));
    }
    let total_weight = weights.iter().sum::<usize>();
    if total_weight == 0 {
        return Err(GmrfError::DimensionMismatch(
            "variance batch weights must sum to a positive value",
        ));
    }

    let mut out = Vector::zeros(dimension);
    let total_weight = total_weight as f64;
    for (vector, weight) in vectors.iter().zip(weights.iter().copied()) {
        let scale = weight as f64 / total_weight;
        for i in 0..dimension {
            out[i] += scale * vector[i];
        }
    }
    Ok(out)
}

/// Stabilize a vector of variance estimates according to `floor`.
pub fn stabilize_variances(
    variances: &Vector,
    floor: VarianceFloor,
) -> Result<StabilizedVariance, GmrfError> {
    if variances.iter().any(|value| !value.is_finite()) {
        return Err(GmrfError::NumericalInstability(
            "variance vector contains non-finite entries",
        ));
    }

    let floor_value = match floor {
        VarianceFloor::Zero => 0.0,
        VarianceFloor::PositiveMean { scale } => {
            if !scale.is_finite() || scale < 0.0 {
                return Err(GmrfError::NumericalInstability(
                    "variance floor scale must be finite and nonnegative",
                ));
            }
            let positive_sum = variances
                .iter()
                .copied()
                .filter(|value| *value > 0.0)
                .sum::<f64>();
            let positive_count = variances.iter().filter(|value| **value > 0.0).count();
            let positive_mean = if positive_count > 0 {
                positive_sum / positive_count as f64
            } else {
                1.0
            };
            positive_mean.abs().max(1.0) * scale
        }
    };

    let mut floor_hits = 0_usize;
    let values = Vector::from_iterator(
        variances.len(),
        variances.iter().copied().map(|value| {
            if value > floor_value {
                value
            } else {
                floor_hits += 1;
                floor_value
            }
        }),
    );

    Ok(StabilizedVariance {
        values,
        floor: floor_value,
        floor_hits,
    })
}

/// Stabilize `unconstrained - removed` variance estimates.
pub fn stabilize_removed_variances(
    unconstrained: &Vector,
    removed: &Vector,
    floor: VarianceFloor,
) -> Result<StabilizedVariance, GmrfError> {
    if unconstrained.len() != removed.len() {
        return Err(GmrfError::DimensionMismatch(
            "variance vectors must have matching dimensions",
        ));
    }
    let raw = Vector::from_iterator(
        unconstrained.len(),
        (0..unconstrained.len()).map(|i| unconstrained[i] - removed[i]),
    );
    stabilize_variances(&raw, floor)
}

/// Clamp a posterior variance estimate into `[0, prior]` pointwise.
pub fn clip_vector_to_prior(prior: &Vector, posterior: &Vector) -> Result<Vector, GmrfError> {
    if prior.len() != posterior.len() {
        return Err(GmrfError::DimensionMismatch(
            "prior and posterior variance vectors must have matching dimensions",
        ));
    }
    Ok(Vector::from_iterator(
        prior.len(),
        (0..prior.len()).map(|i| posterior[i].max(0.0).min(prior[i].max(0.0))),
    ))
}

fn normalize_pair(row: usize, col: usize) -> (usize, usize) {
    if row >= col {
        (row, col)
    } else {
        (col, row)
    }
}

fn draw_probe<R: Rng + ?Sized>(
    dimension: usize,
    distribution: ProbeDistribution,
    rng: &mut R,
) -> Vector {
    Vector::from_fn(dimension, |_| match distribution {
        ProbeDistribution::Gaussian => rng.sample(StandardNormal),
        ProbeDistribution::Rademacher => {
            if rng.gen_bool(0.5) {
                1.0
            } else {
                -1.0
            }
        }
    })
}

fn finalize_variance_estimate(
    estimator: VarianceEstimator,
    batch_estimates: Vec<Vector>,
    batch_sizes: Vec<usize>,
) -> Result<VarianceEstimate, GmrfError> {
    let sample_count = batch_sizes.iter().sum::<usize>();
    let values = weighted_average_vectors(&batch_estimates, &batch_sizes)?;
    let mut num_negative = 0_usize;
    let mut min_value = f64::INFINITY;
    for value in values.iter().copied() {
        if value < 0.0 {
            num_negative += 1;
        }
        min_value = min_value.min(value);
    }
    if values.is_empty() {
        min_value = 0.0;
    }

    let (batch_standard_error, relative_standard_error) = if batch_estimates.len() > 1 {
        let batch_count = batch_estimates.len();
        let mut variance = Vector::zeros(values.len());
        for batch in &batch_estimates {
            for i in 0..values.len() {
                let diff = batch[i] - values[i];
                variance[i] += diff * diff;
            }
        }
        let denom = (batch_count - 1) as f64;
        let se = Vector::from_iterator(
            values.len(),
            (0..values.len()).map(|i| (variance[i] / denom / batch_count as f64).sqrt()),
        );
        let rel = Vector::from_iterator(
            values.len(),
            (0..values.len()).map(|i| {
                let scale = values[i].abs();
                if scale > 1e-15 {
                    se[i] / scale
                } else {
                    0.0
                }
            }),
        );
        (Some(se), Some(rel))
    } else {
        (None, None)
    };

    Ok(VarianceEstimate {
        values,
        batch_standard_error,
        relative_standard_error,
        num_negative,
        min_value,
        estimator,
        sample_count,
        batch_sizes,
    })
}

fn finalize_weighted_trace_estimate(
    estimator: VarianceEstimator,
    batch_estimates: Vec<f64>,
    batch_sizes: Vec<usize>,
) -> Result<WeightedTraceEstimate, GmrfError> {
    if batch_estimates.is_empty() || batch_estimates.len() != batch_sizes.len() {
        return Err(GmrfError::DimensionMismatch(
            "trace batches and weights must be non-empty and aligned",
        ));
    }
    if batch_estimates.iter().any(|value| !value.is_finite()) {
        return Err(GmrfError::NumericalInstability(
            "weighted trace estimate contains non-finite entries",
        ));
    }
    let sample_count = batch_sizes.iter().sum::<usize>();
    if sample_count == 0 {
        return Err(GmrfError::DimensionMismatch(
            "weighted trace batch sizes must sum to a positive value",
        ));
    }

    let total_weight = sample_count as f64;
    let value = batch_estimates
        .iter()
        .zip(batch_sizes.iter().copied())
        .map(|(estimate, size)| *estimate * size as f64 / total_weight)
        .sum::<f64>();

    let (batch_standard_error, relative_standard_error) = if batch_estimates.len() > 1 {
        let batch_count = batch_estimates.len();
        let variance = batch_estimates
            .iter()
            .map(|estimate| {
                let diff = *estimate - value;
                diff * diff
            })
            .sum::<f64>()
            / (batch_count - 1) as f64;
        let se = (variance / batch_count as f64).sqrt();
        let rel = if value.abs() > 1e-15 {
            se / value.abs()
        } else {
            0.0
        };
        (Some(se), Some(rel))
    } else {
        (None, None)
    };

    Ok(WeightedTraceEstimate {
        value,
        batch_estimates,
        batch_sizes,
        batch_standard_error,
        relative_standard_error,
        estimator,
        sample_count,
    })
}

/// Exact `tr(W Q^-1)` for a sparse weight matrix `W`.
pub fn exact_weighted_covariance_trace(
    factor: &SparseCholeskyFactor,
    weight: &SparseMatrix,
) -> Result<WeightedTraceEstimate, GmrfError> {
    let dimension = factor.dimension();
    if weight.nrows() != dimension || weight.ncols() != dimension {
        return Err(GmrfError::DimensionMismatch(
            "weighted trace matrix dimensions must match factor dimension",
        ));
    }

    let mut entries_by_col = BTreeMap::<usize, Vec<(usize, f64)>>::new();
    for (row, col, value) in weight.triplet_iter() {
        if !value.is_finite() {
            return Err(GmrfError::NumericalInstability(
                "weighted trace matrix contains non-finite entries",
            ));
        }
        if *value != 0.0 {
            entries_by_col.entry(col).or_default().push((row, *value));
        }
    }

    let mut trace = 0.0;
    for (col, entries) in entries_by_col {
        let mut rhs = Vector::zeros(dimension);
        rhs[col] = 1.0;
        let solved = factor.solve(&rhs)?;
        for (row, value) in entries {
            trace += value * solved[row];
        }
    }

    finalize_weighted_trace_estimate(VarianceEstimator::ExactSolves, vec![trace], vec![1])
}

/// Deterministic batched Hutchinson estimate of `tr(W Q^-1)`.
pub fn estimate_hutchinson_weighted_covariance_trace(
    factor: &SparseCholeskyFactor,
    weight: &SparseMatrix,
    config: ProbeBatchConfig,
    distribution: ProbeDistribution,
) -> Result<WeightedTraceEstimate, GmrfError> {
    let dimension = factor.dimension();
    if weight.nrows() != dimension || weight.ncols() != dimension {
        return Err(GmrfError::DimensionMismatch(
            "weighted trace matrix dimensions must match factor dimension",
        ));
    }
    if weight
        .triplet_iter()
        .any(|(_, _, value)| !value.is_finite())
    {
        return Err(GmrfError::NumericalInstability(
            "weighted trace matrix contains non-finite entries",
        ));
    }

    let batch_sizes = probe_batch_sizes(config.num_probes, config.batch_count)?;
    let mut batches = Vec::with_capacity(batch_sizes.len());
    for (batch_index, batch_size) in batch_sizes.iter().copied().enumerate() {
        let mut rng =
            rand::rngs::StdRng::seed_from_u64(probe_batch_seed(config.rng_seed, batch_index));
        let mut estimate = 0.0;
        for _ in 0..batch_size {
            let probe = draw_probe(dimension, distribution, &mut rng);
            let solved = factor.solve(&probe)?;
            let weighted_solved = weight.mul_vec(&solved);
            estimate += probe.dot(&weighted_solved);
        }
        batches.push(estimate / batch_size as f64);
    }

    finalize_weighted_trace_estimate(VarianceEstimator::Hutchinson, batches, batch_sizes)
}

/// Exact `tr(diag(output_weights) A Q^-1 A^T)` for a sparse row operator `A`.
pub fn exact_weighted_transformed_covariance_trace(
    factor: &SparseCholeskyFactor,
    operator: &SparseRowOperator,
    output_weights: &Vector,
) -> Result<WeightedTraceEstimate, GmrfError> {
    if output_weights.len() != operator.nrows() {
        return Err(GmrfError::DimensionMismatch(
            "output weights length must match transformed output dimension",
        ));
    }
    if output_weights.iter().any(|value| !value.is_finite()) {
        return Err(GmrfError::NumericalInstability(
            "output weights contain non-finite entries",
        ));
    }
    let variances = exact_solve_transformed_diag(factor, operator)?.values;
    let trace = variances.dot(output_weights);
    finalize_weighted_trace_estimate(VarianceEstimator::ExactSolves, vec![trace], vec![1])
}

/// Deterministic batched Hutchinson estimate of
/// `tr(diag(output_weights) A Q^-1 A^T)`.
pub fn estimate_hutchinson_weighted_transformed_covariance_trace(
    factor: &SparseCholeskyFactor,
    operator: &SparseRowOperator,
    output_weights: &Vector,
    config: ProbeBatchConfig,
    distribution: ProbeDistribution,
) -> Result<WeightedTraceEstimate, GmrfError> {
    if operator.ncols != factor.dimension() {
        return Err(GmrfError::DimensionMismatch(
            "operator columns must match factor dimension",
        ));
    }
    if output_weights.len() != operator.nrows() {
        return Err(GmrfError::DimensionMismatch(
            "output weights length must match transformed output dimension",
        ));
    }
    if output_weights.iter().any(|value| !value.is_finite()) {
        return Err(GmrfError::NumericalInstability(
            "output weights contain non-finite entries",
        ));
    }

    let batch_sizes = probe_batch_sizes(config.num_probes, config.batch_count)?;
    let mut batches = Vec::with_capacity(batch_sizes.len());
    for (batch_index, batch_size) in batch_sizes.iter().copied().enumerate() {
        let mut rng =
            rand::rngs::StdRng::seed_from_u64(probe_batch_seed(config.rng_seed, batch_index));
        let mut estimate = 0.0;
        for _ in 0..batch_size {
            let probe = draw_probe(operator.nrows(), distribution, &mut rng);
            let rhs = operator.apply_transpose(&probe)?;
            let weighted_probe = probe.component_mul(output_weights);
            let solved = factor.solve(&rhs)?;
            let projected = operator.apply(&solved)?;
            estimate += weighted_probe.dot(&projected);
        }
        batches.push(estimate / batch_size as f64);
    }

    finalize_weighted_trace_estimate(VarianceEstimator::Hutchinson, batches, batch_sizes)
}

/// Exact transformed marginal variances and `tr(diag(output_weights) A Q^-1 A^T)`.
///
/// This computes the transformed diagonal once and obtains the weighted trace as
/// a dot product with `output_weights`.
pub fn exact_transformed_variance_weighted_trace(
    factor: &SparseCholeskyFactor,
    operator: &SparseRowOperator,
    output_weights: &Vector,
) -> Result<TransformedVarianceWeightedTraceEstimate, GmrfError> {
    validate_transformed_trace_inputs(operator, output_weights)?;
    let variances = exact_solve_transformed_diag(factor, operator)?;
    let trace = variances.values.dot(output_weights);
    let weighted_trace =
        finalize_weighted_trace_estimate(VarianceEstimator::ExactSolves, vec![trace], vec![1])?;
    Ok(TransformedVarianceWeightedTraceEstimate {
        variance_batch_estimates: vec![variances.values.clone()],
        variances,
        weighted_trace,
        floor_hits: 0,
    })
}

/// Dispatch exact or Hutchinson transformed variances for a factored precision.
pub fn estimate_factored_transformed_variances(
    factor: &SparseCholeskyFactor,
    operator: &SparseRowOperator,
    mode: TransformedVarianceMode,
) -> Result<VarianceEstimate, GmrfError> {
    match mode {
        TransformedVarianceMode::Exact => exact_solve_transformed_diag(factor, operator),
        TransformedVarianceMode::Hutchinson {
            config,
            floor,
            distribution,
        } => estimate_factored_hutchinson_transformed_variances(
            factor,
            operator,
            config,
            floor,
            distribution,
        ),
        TransformedVarianceMode::Auto {
            exact_max_dofs,
            config,
            floor,
            distribution,
        } => {
            if factor.dimension() <= exact_max_dofs {
                exact_solve_transformed_diag(factor, operator)
            } else {
                estimate_factored_hutchinson_transformed_variances(
                    factor,
                    operator,
                    config,
                    floor,
                    distribution,
                )
            }
        }
    }
}

fn estimate_factored_hutchinson_transformed_variances(
    factor: &SparseCholeskyFactor,
    operator: &SparseRowOperator,
    config: ProbeBatchConfig,
    floor: VarianceFloor,
    distribution: ProbeDistribution,
) -> Result<VarianceEstimate, GmrfError> {
    if operator.ncols != factor.dimension() {
        return Err(GmrfError::DimensionMismatch(
            "operator columns must match factor dimension",
        ));
    }
    let batch_sizes = probe_batch_sizes(config.num_probes, config.batch_count)?;
    let operator_tilde = operator.permute_columns_to_factor(&factor.permutation())?;
    let mut batch_estimates = Vec::with_capacity(batch_sizes.len());
    for (batch_index, batch_size) in batch_sizes.iter().copied().enumerate() {
        let mut rng =
            rand::rngs::StdRng::seed_from_u64(probe_batch_seed(config.rng_seed, batch_index));
        let mut raw = Vector::zeros(operator.nrows());
        for _ in 0..batch_size {
            let probe = draw_probe(operator.nrows(), distribution, &mut rng);
            let rhs_tilde = operator_tilde.apply_transpose(&probe)?;
            let solved_tilde = factor.solve_permuted(&rhs_tilde)?;
            let projected = operator_tilde.apply(&solved_tilde)?;
            raw += projected.component_mul(&probe);
        }
        let stabilized = stabilize_variances(&(raw / batch_size as f64), floor)?;
        batch_estimates.push(stabilized.values);
    }
    finalize_variance_estimate(VarianceEstimator::Hutchinson, batch_estimates, batch_sizes)
}

/// Dispatch exact or Hutchinson transformed variances and a weighted trace for a factored precision.
pub fn estimate_factored_transformed_variance_weighted_trace(
    factor: &SparseCholeskyFactor,
    operator: &SparseRowOperator,
    output_weights: &Vector,
    mode: TransformedVarianceMode,
) -> Result<TransformedVarianceWeightedTraceEstimate, GmrfError> {
    match mode {
        TransformedVarianceMode::Exact => {
            exact_transformed_variance_weighted_trace(factor, operator, output_weights)
        }
        TransformedVarianceMode::Hutchinson {
            config,
            floor,
            distribution,
        } => estimate_hutchinson_transformed_variance_weighted_trace(
            factor,
            operator,
            output_weights,
            config,
            floor,
            distribution,
        ),
        TransformedVarianceMode::Auto {
            exact_max_dofs,
            config,
            floor,
            distribution,
        } => {
            if factor.dimension() <= exact_max_dofs {
                exact_transformed_variance_weighted_trace(factor, operator, output_weights)
            } else {
                estimate_hutchinson_transformed_variance_weighted_trace(
                    factor,
                    operator,
                    output_weights,
                    config,
                    floor,
                    distribution,
                )
            }
        }
    }
}

/// Dispatch exact or Hutchinson transformed variances through a constrained covariance action.
pub fn estimate_constrained_transformed_variances(
    solver: &ConstrainedPrecisionSolver,
    operator: &SparseRowOperator,
    mode: TransformedVarianceMode,
) -> Result<VarianceEstimate, GmrfError> {
    match mode {
        TransformedVarianceMode::Exact => {
            let values = solver.exact_transformed_variances(operator)?;
            finalize_variance_estimate(VarianceEstimator::ExactSolves, vec![values], vec![1])
        }
        TransformedVarianceMode::Hutchinson {
            config,
            floor,
            distribution,
        } => estimate_constrained_hutchinson_transformed_variances(
            solver,
            operator,
            config,
            floor,
            distribution,
        ),
        TransformedVarianceMode::Auto {
            exact_max_dofs,
            config,
            floor,
            distribution,
        } => {
            if solver.latent_dim() <= exact_max_dofs {
                let values = solver.exact_transformed_variances(operator)?;
                finalize_variance_estimate(VarianceEstimator::ExactSolves, vec![values], vec![1])
            } else {
                estimate_constrained_hutchinson_transformed_variances(
                    solver,
                    operator,
                    config,
                    floor,
                    distribution,
                )
            }
        }
    }
}

fn estimate_constrained_hutchinson_transformed_variances(
    solver: &ConstrainedPrecisionSolver,
    operator: &SparseRowOperator,
    config: ProbeBatchConfig,
    floor: VarianceFloor,
    distribution: ProbeDistribution,
) -> Result<VarianceEstimate, GmrfError> {
    if operator.ncols != solver.latent_dim() {
        return Err(GmrfError::DimensionMismatch(
            "operator columns must match constrained latent dimension",
        ));
    }
    let batch_sizes = probe_batch_sizes(config.num_probes, config.batch_count)?;
    let mut batch_estimates = Vec::with_capacity(batch_sizes.len());
    for (batch_index, batch_size) in batch_sizes.iter().copied().enumerate() {
        let mut rng =
            rand::rngs::StdRng::seed_from_u64(probe_batch_seed(config.rng_seed, batch_index));
        let mut raw = Vector::zeros(operator.nrows());
        for _ in 0..batch_size {
            let probe = draw_probe(operator.nrows(), distribution, &mut rng);
            let rhs = operator.apply_transpose(&probe)?;
            let solved = solver.solve_covariance_action(&rhs)?;
            let projected = operator.apply(&solved)?;
            raw += projected.component_mul(&probe);
        }
        let stabilized = stabilize_variances(&(raw / batch_size as f64), floor)?;
        batch_estimates.push(stabilized.values);
    }
    finalize_variance_estimate(VarianceEstimator::Hutchinson, batch_estimates, batch_sizes)
}

/// Deterministic Hutchinson estimate of transformed marginal variances and their weighted trace.
///
/// Each stochastic probe is solved once. The weighted trace batches are computed
/// from the same stabilized transformed variance batches returned in `variances`.
pub fn estimate_hutchinson_transformed_variance_weighted_trace(
    factor: &SparseCholeskyFactor,
    operator: &SparseRowOperator,
    output_weights: &Vector,
    config: ProbeBatchConfig,
    floor: VarianceFloor,
    distribution: ProbeDistribution,
) -> Result<TransformedVarianceWeightedTraceEstimate, GmrfError> {
    if operator.ncols != factor.dimension() {
        return Err(GmrfError::DimensionMismatch(
            "operator columns must match factor dimension",
        ));
    }
    estimate_hutchinson_transformed_variance_weighted_trace_with_solve(
        operator,
        output_weights,
        config,
        floor,
        distribution,
        |rhs| factor.solve(rhs),
    )
}

/// Deterministic Hutchinson estimate of transformed variances and weighted trace
/// with a custom covariance action.
pub fn estimate_hutchinson_transformed_variance_weighted_trace_with_solve<F>(
    operator: &SparseRowOperator,
    output_weights: &Vector,
    config: ProbeBatchConfig,
    floor: VarianceFloor,
    distribution: ProbeDistribution,
    mut solve_covariance_action: F,
) -> Result<TransformedVarianceWeightedTraceEstimate, GmrfError>
where
    F: FnMut(&Vector) -> Result<Vector, GmrfError>,
{
    validate_transformed_trace_inputs(operator, output_weights)?;

    let batch_sizes = probe_batch_sizes(config.num_probes, config.batch_count)?;
    let mut variance_batches = Vec::with_capacity(batch_sizes.len());
    let mut trace_batches = Vec::with_capacity(batch_sizes.len());
    let mut floor_hits = 0_usize;

    for (batch_index, batch_size) in batch_sizes.iter().copied().enumerate() {
        let mut rng =
            rand::rngs::StdRng::seed_from_u64(probe_batch_seed(config.rng_seed, batch_index));
        let mut batch = Vector::zeros(operator.nrows());
        for _ in 0..batch_size {
            let probe = draw_probe(operator.nrows(), distribution, &mut rng);
            let rhs = operator.apply_transpose(&probe)?;
            let solved = solve_covariance_action(&rhs)?;
            let projected = operator.apply(&solved)?;
            batch += projected.component_mul(&probe);
        }
        batch = batch / (batch_size as f64);
        let stabilized = stabilize_variances(&batch, floor)?;
        floor_hits += stabilized.floor_hits;
        trace_batches.push(stabilized.values.dot(output_weights));
        variance_batches.push(stabilized.values);
    }

    let variance_batch_estimates = variance_batches.clone();
    let variances = finalize_variance_estimate(
        VarianceEstimator::Hutchinson,
        variance_batches,
        batch_sizes.clone(),
    )?;
    let weighted_trace = finalize_weighted_trace_estimate(
        VarianceEstimator::Hutchinson,
        trace_batches,
        batch_sizes,
    )?;

    Ok(TransformedVarianceWeightedTraceEstimate {
        variances,
        variance_batch_estimates,
        weighted_trace,
        floor_hits,
    })
}

fn validate_transformed_trace_inputs(
    operator: &SparseRowOperator,
    output_weights: &Vector,
) -> Result<(), GmrfError> {
    if output_weights.len() != operator.nrows() {
        return Err(GmrfError::DimensionMismatch(
            "output weights length must match transformed output dimension",
        ));
    }
    if output_weights.iter().any(|value| !value.is_finite()) {
        return Err(GmrfError::NumericalInstability(
            "output weights contain non-finite entries",
        ));
    }
    Ok(())
}

fn resolve_contiguous_blocks(
    dimension: usize,
    block_size: usize,
) -> Result<Vec<Vec<usize>>, GmrfError> {
    if block_size < 2 {
        return Err(GmrfError::DimensionMismatch(
            "local RB contiguous block_size must be at least 2",
        ));
    }
    let mut blocks = Vec::new();
    let mut start = 0;
    while start < dimension {
        let end = (start + block_size).min(dimension);
        blocks.push((start..end).collect());
        start = end;
    }
    Ok(blocks)
}

type ResolvedBlocks = (Vec<Vec<usize>>, Option<Vec<usize>>);

fn resolve_blocks(mode: &LatentBlockMode, dimension: usize) -> Result<ResolvedBlocks, GmrfError> {
    match mode {
        LatentBlockMode::ContiguousPermuted { block_size } => {
            Ok((resolve_contiguous_blocks(dimension, *block_size)?, None))
        }
        LatentBlockMode::Explicit {
            blocks,
            row_assignments,
        } => {
            let mut resolved = Vec::with_capacity(blocks.len());
            for block in blocks {
                if block.is_empty() {
                    return Err(GmrfError::DimensionMismatch(
                        "local RB blocks must be non-empty",
                    ));
                }
                let mut current = Vec::with_capacity(block.len());
                for index in block {
                    if index.0 >= dimension {
                        return Err(GmrfError::DimensionMismatch(
                            "local RB block index exceeds dimension",
                        ));
                    }
                    current.push(index.0);
                }
                current.sort_unstable();
                current.dedup();
                resolved.push(current);
            }
            let assignments = row_assignments
                .as_ref()
                .map(|assignments| assignments.iter().map(|id| id.0).collect::<Vec<_>>());
            if let Some(assignments) = &assignments {
                if assignments.iter().any(|id| *id >= resolved.len()) {
                    return Err(GmrfError::DimensionMismatch(
                        "local RB row assignment references an unknown block",
                    ));
                }
            }
            Ok((resolved, assignments))
        }
    }
}

fn validate_latent_partition(blocks: &[Vec<usize>], dimension: usize) -> Result<(), GmrfError> {
    let mut seen = vec![false; dimension];
    for block in blocks {
        for index in block {
            if seen[*index] {
                return Err(GmrfError::DimensionMismatch(
                    "latent local RB blocks must form a partition",
                ));
            }
            seen[*index] = true;
        }
    }
    if seen.iter().any(|flag| !*flag) {
        return Err(GmrfError::DimensionMismatch(
            "latent local RB blocks must cover every coordinate",
        ));
    }
    Ok(())
}

fn block_index_map(block: &[usize]) -> BTreeMap<usize, usize> {
    block
        .iter()
        .copied()
        .enumerate()
        .map(|(local, global)| (global, local))
        .collect()
}

fn dense_submatrix(matrix: &SparseMatrix, block: &[usize]) -> DenseMatrix {
    let index_map = block_index_map(block);
    let mut dense = DenseMatrix::zeros(block.len(), block.len());
    for (row, col, value) in matrix.triplet_iter() {
        if let (Some(local_row), Some(local_col)) = (index_map.get(&row), index_map.get(&col)) {
            dense[(*local_row, *local_col)] += *value;
        }
    }
    dense
}

fn dense_inverse(matrix: &DenseMatrix) -> Result<DenseMatrix, GmrfError> {
    let factor = matrix
        .llt(Side::Lower)
        .map_err(|_| GmrfError::NonPositiveDefinite)?;
    let n = matrix.nrows();
    let mut inverse = DenseMatrix::identity(n, n);
    factor.solve_in_place(inverse.as_mut());
    Ok(inverse)
}

fn dense_matvec_block(matrix: &DenseMatrix, rhs: &[f64]) -> Vector {
    Vector::from_fn(matrix.nrows(), |row| {
        (0..matrix.ncols())
            .map(|col| matrix[(row, col)] * rhs[col])
            .sum::<f64>()
    })
}

fn rows_by_assignment(
    operator: &SparseRowOperator,
    blocks: &[Vec<usize>],
    explicit_assignments: Option<&[usize]>,
) -> Result<Vec<Vec<usize>>, GmrfError> {
    if let Some(assignments) = explicit_assignments {
        if assignments.len() != operator.nrows() {
            return Err(GmrfError::DimensionMismatch(
                "local RB row assignments must match operator row count",
            ));
        }
        let mut rows = vec![Vec::new(); blocks.len()];
        for (row_index, block_id) in assignments.iter().copied().enumerate() {
            if block_id >= blocks.len() {
                return Err(GmrfError::DimensionMismatch(
                    "local RB row assignment references an unknown block",
                ));
            }
            let block_set = blocks[block_id].iter().copied().collect::<BTreeSet<_>>();
            if operator.rows[row_index]
                .iter()
                .any(|(col, _)| !block_set.contains(col))
            {
                return Err(GmrfError::DimensionMismatch(
                    "transformed local RB row support is not contained in its assigned block",
                ));
            }
            rows[block_id].push(row_index);
        }
        return Ok(rows);
    }

    let block_sets = blocks
        .iter()
        .map(|block| block.iter().copied().collect::<BTreeSet<_>>())
        .collect::<Vec<_>>();
    let mut rows = vec![Vec::new(); blocks.len()];
    for (row_index, row) in operator.rows.iter().enumerate() {
        let containing = block_sets
            .iter()
            .enumerate()
            .filter(|(_, block)| row.iter().all(|(col, _)| block.contains(col)))
            .map(|(block_id, _)| block_id)
            .collect::<Vec<_>>();
        if containing.len() != 1 {
            return Err(GmrfError::DimensionMismatch(
                "each transformed local RB row must be contained in exactly one block",
            ));
        }
        rows[containing[0]].push(row_index);
    }
    Ok(rows)
}

/// Estimate `diag(Q^-1)` with deterministic batched Hutchinson probes.
pub fn estimate_batched_hutchinson_variances(
    gmrf: &mut Gmrf,
    config: ProbeBatchConfig,
    floor: VarianceFloor,
) -> Result<BatchedVarianceEstimate, GmrfError> {
    let batch_sizes = probe_batch_sizes(config.num_probes, config.batch_count)?;
    let mut batch_estimates = Vec::with_capacity(batch_sizes.len());
    let mut floor_hits = 0_usize;

    for (batch_index, batch_size) in batch_sizes.iter().copied().enumerate() {
        let raw = estimate_hutchinson_variances(
            gmrf,
            batch_size,
            1,
            probe_batch_seed(config.rng_seed, batch_index),
            ProbeDistribution::Rademacher,
        )?
        .values;
        let stabilized = stabilize_variances(&raw, floor)?;
        floor_hits += stabilized.floor_hits;
        batch_estimates.push(stabilized.values);
    }

    let estimate = weighted_average_vectors(&batch_estimates, &batch_sizes)?;
    Ok(BatchedVarianceEstimate {
        estimate,
        batch_estimates,
        batch_sizes,
        floor_hits,
    })
}

/// Estimate marginal variances from constrained Monte Carlo samples.
pub fn estimate_constrained_mc_variances<R: Rng + ?Sized>(
    gmrf: &mut Gmrf,
    constraint_matrix: &DenseMatrix,
    constraint_rhs: &Vector,
    num_samples: usize,
    rng: &mut R,
) -> Result<Vector, GmrfError> {
    if num_samples == 0 {
        return Err(GmrfError::DimensionMismatch(
            "at least one sample is required",
        ));
    }
    if constraint_matrix.nrows() == 0 {
        return gmrf.mc_variances(num_samples, rng);
    }

    let constrained_mean = gmrf.constrained_mean(constraint_matrix, constraint_rhs)?;
    let dim = gmrf.dimension();
    let mut variances = Vector::zeros(dim);
    for _ in 0..num_samples {
        let draw = gmrf.sample_constrained(constraint_matrix, constraint_rhs, rng)?;
        let centered = &draw - &constrained_mean;
        variances += centered.component_mul(&centered);
    }
    Ok(variances / (num_samples as f64))
}

/// Estimate transformed variances from Monte Carlo samples drawn by `sample_draw`.
pub fn estimate_transformed_mc_variances<R, F>(
    operator: &SparseRowOperator,
    mean: &Vector,
    num_samples: usize,
    rng: &mut R,
    mut sample_draw: F,
) -> Result<Vector, GmrfError>
where
    R: Rng + ?Sized,
    F: FnMut(&mut R) -> Result<Vector, GmrfError>,
{
    if num_samples == 0 {
        return Err(GmrfError::DimensionMismatch(
            "at least one sample is required",
        ));
    }

    let transformed_mean = operator.apply(mean)?;
    let output_dim = operator.nrows();
    let mut variances = Vector::zeros(output_dim);
    for _ in 0..num_samples {
        let draw = sample_draw(rng)?;
        let transformed = operator.apply(&draw)?;
        let centered = &transformed - &transformed_mean;
        variances += centered.component_mul(&centered);
    }
    Ok(variances / (num_samples as f64))
}

/// Estimate latent marginal variances using Gaussian posterior Monte Carlo samples.
pub fn estimate_monte_carlo_variances(
    gmrf: &mut Gmrf,
    num_samples: usize,
    batch_count: usize,
    rng_seed: u64,
) -> Result<VarianceEstimate, GmrfError> {
    let batch_sizes = probe_batch_sizes(num_samples, batch_count)?;
    let dim = gmrf.dimension();
    let mut batches = Vec::with_capacity(batch_sizes.len());

    for (batch_index, batch_size) in batch_sizes.iter().copied().enumerate() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(probe_batch_seed(rng_seed, batch_index));
        let mut variances = Vector::zeros(dim);
        if let Some(factor) = gmrf.precision_factor() {
            for _ in 0..batch_size {
                let mut draw_tilde = Vector::from_fn(dim, |_| rng.sample(StandardNormal));
                factor.solve_l_transpose_permuted_in_place(&mut draw_tilde)?;
                let draw = factor.permute_factor_to_original(&draw_tilde)?;
                variances += draw.component_mul(&draw);
            }
        } else {
            let mean = gmrf.mean().clone();
            for _ in 0..batch_size {
                let draw = gmrf.sample(&mut rng)?;
                let centered = &draw - &mean;
                variances += centered.component_mul(&centered);
            }
        }
        batches.push(variances / batch_size as f64);
    }

    finalize_variance_estimate(VarianceEstimator::MonteCarlo, batches, batch_sizes)
}

/// Estimate constrained latent marginal variances in deterministic batches.
pub fn estimate_monte_carlo_constrained_variances(
    gmrf: &mut Gmrf,
    constraint_matrix: &DenseMatrix,
    constraint_rhs: &Vector,
    num_samples: usize,
    batch_count: usize,
    rng_seed: u64,
) -> Result<VarianceEstimate, GmrfError> {
    if constraint_matrix.nrows() == 0 {
        return estimate_monte_carlo_variances(gmrf, num_samples, batch_count, rng_seed);
    }
    let batch_sizes = probe_batch_sizes(num_samples, batch_count)?;
    let mean = gmrf.constrained_mean(constraint_matrix, constraint_rhs)?;
    let dim = gmrf.dimension();
    let mut batches = Vec::with_capacity(batch_sizes.len());
    for (batch_index, batch_size) in batch_sizes.iter().copied().enumerate() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(probe_batch_seed(rng_seed, batch_index));
        let mut variances = Vector::zeros(dim);
        for _ in 0..batch_size {
            let draw = gmrf.sample_constrained(constraint_matrix, constraint_rhs, &mut rng)?;
            let centered = &draw - &mean;
            variances += centered.component_mul(&centered);
        }
        batches.push(variances / batch_size as f64);
    }
    finalize_variance_estimate(VarianceEstimator::MonteCarlo, batches, batch_sizes)
}

/// Estimate transformed variances using Gaussian posterior Monte Carlo samples.
pub fn estimate_monte_carlo_transformed_variances(
    gmrf: &mut Gmrf,
    operator: &SparseRowOperator,
    num_samples: usize,
    batch_count: usize,
    rng_seed: u64,
) -> Result<VarianceEstimate, GmrfError> {
    if operator.ncols != gmrf.dimension() {
        return Err(GmrfError::DimensionMismatch(
            "operator columns must match latent dimension",
        ));
    }
    let batch_sizes = probe_batch_sizes(num_samples, batch_count)?;
    let mut batches = Vec::with_capacity(batch_sizes.len());
    let dim = gmrf.dimension();

    for (batch_index, batch_size) in batch_sizes.iter().copied().enumerate() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(probe_batch_seed(rng_seed, batch_index));
        let mut variances = Vector::zeros(operator.nrows());
        if let Some(factor) = gmrf.precision_factor() {
            let operator_tilde = operator.permute_columns_to_factor(&factor.permutation())?;
            for _ in 0..batch_size {
                let mut draw_tilde = Vector::from_fn(dim, |_| rng.sample(StandardNormal));
                factor.solve_l_transpose_permuted_in_place(&mut draw_tilde)?;
                let transformed = operator_tilde.apply(&draw_tilde)?;
                variances += transformed.component_mul(&transformed);
            }
        } else {
            let mean = gmrf.mean().clone();
            let transformed_mean = operator.apply(&mean)?;
            for _ in 0..batch_size {
                let draw = gmrf.sample(&mut rng)?;
                let transformed = operator.apply(&draw)?;
                let centered = &transformed - &transformed_mean;
                variances += centered.component_mul(&centered);
            }
        }
        batches.push(variances / batch_size as f64);
    }

    finalize_variance_estimate(VarianceEstimator::MonteCarlo, batches, batch_sizes)
}

/// Estimate constrained transformed marginal variances in deterministic batches.
pub fn estimate_monte_carlo_constrained_transformed_variances(
    gmrf: &mut Gmrf,
    operator: &SparseRowOperator,
    constraint_matrix: &DenseMatrix,
    constraint_rhs: &Vector,
    num_samples: usize,
    batch_count: usize,
    rng_seed: u64,
) -> Result<VarianceEstimate, GmrfError> {
    if operator.ncols != gmrf.dimension() {
        return Err(GmrfError::DimensionMismatch(
            "operator columns must match latent dimension",
        ));
    }
    if constraint_matrix.nrows() == 0 {
        return estimate_monte_carlo_transformed_variances(
            gmrf,
            operator,
            num_samples,
            batch_count,
            rng_seed,
        );
    }
    let batch_sizes = probe_batch_sizes(num_samples, batch_count)?;
    let mean = gmrf.constrained_mean(constraint_matrix, constraint_rhs)?;
    let transformed_mean = operator.apply(&mean)?;
    let mut batches = Vec::with_capacity(batch_sizes.len());
    for (batch_index, batch_size) in batch_sizes.iter().copied().enumerate() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(probe_batch_seed(rng_seed, batch_index));
        let mut variances = Vector::zeros(operator.nrows());
        for _ in 0..batch_size {
            let draw = gmrf.sample_constrained(constraint_matrix, constraint_rhs, &mut rng)?;
            let transformed = operator.apply(&draw)?;
            let centered = &transformed - &transformed_mean;
            variances += centered.component_mul(&centered);
        }
        batches.push(variances / batch_size as f64);
    }
    finalize_variance_estimate(VarianceEstimator::MonteCarlo, batches, batch_sizes)
}

/// Estimate latent marginal variances with Hutchinson diagonal probing.
pub fn estimate_hutchinson_variances(
    gmrf: &mut Gmrf,
    num_probes: usize,
    batch_count: usize,
    rng_seed: u64,
    distribution: ProbeDistribution,
) -> Result<VarianceEstimate, GmrfError> {
    let batch_sizes = probe_batch_sizes(num_probes, batch_count)?;
    let dim = gmrf.dimension();
    let mut batches = Vec::with_capacity(batch_sizes.len());

    for (batch_index, batch_size) in batch_sizes.iter().copied().enumerate() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(probe_batch_seed(rng_seed, batch_index));
        let mut variances = Vector::zeros(dim);
        if let Some(factor) = gmrf.precision_factor() {
            for _ in 0..batch_size {
                let probe_tilde = draw_probe(dim, distribution, &mut rng);
                let solved_tilde = factor.solve_permuted(&probe_tilde)?;
                let estimate_tilde = solved_tilde.component_mul(&probe_tilde);
                let estimate = factor.permute_factor_to_original(&estimate_tilde)?;
                variances += estimate;
            }
        } else {
            for _ in 0..batch_size {
                let probe = draw_probe(dim, distribution, &mut rng);
                let solved = gmrf.solve_precision(&probe)?;
                variances += solved.component_mul(&probe);
            }
        }
        batches.push(variances / batch_size as f64);
    }

    finalize_variance_estimate(VarianceEstimator::Hutchinson, batches, batch_sizes)
}

/// Estimate transformed variances with Hutchinson probing in output space.
pub fn estimate_hutchinson_transformed_variances(
    gmrf: &mut Gmrf,
    operator: &SparseRowOperator,
    num_probes: usize,
    batch_count: usize,
    rng_seed: u64,
    distribution: ProbeDistribution,
) -> Result<VarianceEstimate, GmrfError> {
    if operator.ncols != gmrf.dimension() {
        return Err(GmrfError::DimensionMismatch(
            "operator columns must match latent dimension",
        ));
    }
    let batch_sizes = probe_batch_sizes(num_probes, batch_count)?;
    let mut batches = Vec::with_capacity(batch_sizes.len());

    for (batch_index, batch_size) in batch_sizes.iter().copied().enumerate() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(probe_batch_seed(rng_seed, batch_index));
        let mut variances = Vector::zeros(operator.nrows());
        if let Some(factor) = gmrf.precision_factor() {
            let operator_tilde = operator.permute_columns_to_factor(&factor.permutation())?;
            for _ in 0..batch_size {
                let probe = draw_probe(operator.nrows(), distribution, &mut rng);
                let rhs_tilde = operator_tilde.apply_transpose(&probe)?;
                let solved_tilde = factor.solve_permuted(&rhs_tilde)?;
                let projected = operator_tilde.apply(&solved_tilde)?;
                variances += projected.component_mul(&probe);
            }
        } else {
            for _ in 0..batch_size {
                let probe = draw_probe(operator.nrows(), distribution, &mut rng);
                let rhs = operator.apply_transpose(&probe)?;
                let solved = gmrf.solve_precision(&rhs)?;
                let projected = operator.apply(&solved)?;
                variances += projected.component_mul(&probe);
            }
        }
        batches.push(variances / batch_size as f64);
    }

    finalize_variance_estimate(VarianceEstimator::Hutchinson, batches, batch_sizes)
}

/// Estimate latent marginal variances with local Rao-Blackwellised Gaussian samples.
pub fn estimate_local_rbmc_variances(
    precision: &SparseMatrix,
    factor: &SparseCholeskyFactor,
    block_mode: &LatentBlockMode,
    num_samples: usize,
    batch_count: usize,
    rng_seed: u64,
) -> Result<LocalRbVarianceEstimate, GmrfError> {
    if precision.nrows() != factor.dimension() || precision.ncols() != factor.dimension() {
        return Err(GmrfError::DimensionMismatch(
            "precision and factor dimensions must match",
        ));
    }
    let permutation = factor.permutation();
    let precision_tilde = precision.permute_symmetric(&permutation)?;
    let (blocks, _) = resolve_blocks(block_mode, factor.dimension())?;
    validate_latent_partition(&blocks, factor.dimension())?;

    let inverses = blocks
        .iter()
        .map(|block| dense_inverse(&dense_submatrix(&precision_tilde, block)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut deterministic = Vector::zeros(factor.dimension());
    for (block, inverse) in blocks.iter().zip(&inverses) {
        for (local, global) in block.iter().copied().enumerate() {
            deterministic[global] = inverse[(local, local)];
        }
    }

    let batch_sizes = probe_batch_sizes(num_samples, batch_count)?;
    let mut batches = Vec::with_capacity(batch_sizes.len());
    let mut residual_batches = Vec::with_capacity(batch_sizes.len());
    for (batch_index, batch_size) in batch_sizes.iter().copied().enumerate() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(probe_batch_seed(rng_seed, batch_index));
        let mut residual = Vector::zeros(factor.dimension());
        for _ in 0..batch_size {
            let z = Vector::from_fn(factor.dimension(), |_| rng.sample(StandardNormal));
            let mut delta = z.clone();
            factor.solve_l_transpose_permuted_in_place(&mut delta)?;
            let qdelta = precision_tilde.mul_vec(&delta);
            for (block, inverse) in blocks.iter().zip(&inverses) {
                let rhs = block
                    .iter()
                    .map(|global| qdelta[*global])
                    .collect::<Vec<_>>();
                let y = dense_matvec_block(inverse, &rhs);
                for (local, global) in block.iter().copied().enumerate() {
                    let h = delta[global] - y[local];
                    residual[global] += h * h;
                }
            }
        }
        let residual_estimate = residual / batch_size as f64;
        residual_batches.push(residual_estimate.clone());
        batches.push(&deterministic + residual_estimate);
    }

    let estimate_tilde =
        finalize_variance_estimate(VarianceEstimator::LocalRbmc, batches, batch_sizes)?;
    let residual_tilde = weighted_average_vectors(&residual_batches, &estimate_tilde.batch_sizes)?;
    let values = factor.permute_factor_to_original(&estimate_tilde.values)?;
    let residual = factor.permute_factor_to_original(&residual_tilde)?;
    let deterministic_original = factor.permute_factor_to_original(&deterministic)?;
    let deterministic_fraction = Vector::from_iterator(
        values.len(),
        (0..values.len()).map(|i| {
            if values[i].abs() > 1e-15 {
                deterministic_original[i] / values[i]
            } else {
                0.0
            }
        }),
    );
    let estimate = VarianceEstimate {
        values,
        batch_standard_error: estimate_tilde
            .batch_standard_error
            .as_ref()
            .map(|se| factor.permute_factor_to_original(se))
            .transpose()?,
        relative_standard_error: estimate_tilde
            .relative_standard_error
            .as_ref()
            .map(|se| factor.permute_factor_to_original(se))
            .transpose()?,
        num_negative: estimate_tilde.num_negative,
        min_value: estimate_tilde.min_value,
        estimator: estimate_tilde.estimator,
        sample_count: estimate_tilde.sample_count,
        batch_sizes: estimate_tilde.batch_sizes,
    };

    Ok(LocalRbVarianceEstimate {
        estimate,
        diagnostics: LocalRbDiagnostics {
            deterministic_fraction,
            residual_variance_estimate: residual,
        },
    })
}

/// Estimate transformed variances with local RB and row-assigned blocks.
pub fn estimate_local_rbmc_transformed_variances(
    precision: &SparseMatrix,
    factor: &SparseCholeskyFactor,
    operator: &SparseRowOperator,
    block_mode: &LatentBlockMode,
    num_samples: usize,
    batch_count: usize,
    rng_seed: u64,
) -> Result<LocalRbVarianceEstimate, GmrfError> {
    if operator.ncols != factor.dimension()
        || precision.nrows() != factor.dimension()
        || precision.ncols() != factor.dimension()
    {
        return Err(GmrfError::DimensionMismatch(
            "operator, precision, and factor dimensions must align",
        ));
    }
    let permutation = factor.permutation();
    let precision_tilde = precision.permute_symmetric(&permutation)?;
    let operator_tilde = operator.permute_columns_to_factor(&permutation)?;
    let (blocks, assignments) = resolve_blocks(block_mode, factor.dimension())?;
    let rows_for_block = rows_by_assignment(&operator_tilde, &blocks, assignments.as_deref())?;

    let inverses = blocks
        .iter()
        .map(|block| dense_inverse(&dense_submatrix(&precision_tilde, block)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut deterministic = Vector::zeros(operator.nrows());
    let block_maps = blocks
        .iter()
        .map(|block| block_index_map(block))
        .collect::<Vec<_>>();

    for (block_id, rows) in rows_for_block.iter().enumerate() {
        let inverse = &inverses[block_id];
        let map = &block_maps[block_id];
        for row_index in rows {
            let row = &operator_tilde.rows[*row_index];
            let mut value = 0.0;
            for (col_a, weight_a) in row {
                for (col_b, weight_b) in row {
                    value += *weight_a * *weight_b * inverse[(map[col_a], map[col_b])];
                }
            }
            deterministic[*row_index] = value;
        }
    }

    let batch_sizes = probe_batch_sizes(num_samples, batch_count)?;
    let mut batches = Vec::with_capacity(batch_sizes.len());
    let mut residual_batches = Vec::with_capacity(batch_sizes.len());
    for (batch_index, batch_size) in batch_sizes.iter().copied().enumerate() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(probe_batch_seed(rng_seed, batch_index));
        let mut residual = Vector::zeros(operator.nrows());
        for _ in 0..batch_size {
            let z = Vector::from_fn(factor.dimension(), |_| rng.sample(StandardNormal));
            let mut delta = z.clone();
            factor.solve_l_transpose_permuted_in_place(&mut delta)?;
            let qdelta = precision_tilde.mul_vec(&delta);
            for (block_id, block) in blocks.iter().enumerate() {
                let rows = &rows_for_block[block_id];
                if rows.is_empty() {
                    continue;
                }
                let inverse = &inverses[block_id];
                let map = &block_maps[block_id];
                let rhs = block
                    .iter()
                    .map(|global| qdelta[*global])
                    .collect::<Vec<_>>();
                let y = dense_matvec_block(inverse, &rhs);
                let h = Vector::from_fn(block.len(), |local| delta[block[local]] - y[local]);
                for row_index in rows {
                    let eta = operator_tilde.rows[*row_index]
                        .iter()
                        .map(|(col, weight)| *weight * h[map[col]])
                        .sum::<f64>();
                    residual[*row_index] += eta * eta;
                }
            }
        }
        let residual_estimate = residual / batch_size as f64;
        residual_batches.push(residual_estimate.clone());
        batches.push(&deterministic + residual_estimate);
    }

    let estimate = finalize_variance_estimate(VarianceEstimator::LocalRbmc, batches, batch_sizes)?;
    let residual = weighted_average_vectors(&residual_batches, &estimate.batch_sizes)?;
    let deterministic_fraction = Vector::from_iterator(
        estimate.values.len(),
        (0..estimate.values.len()).map(|i| {
            if estimate.values[i].abs() > 1e-15 {
                deterministic[i] / estimate.values[i]
            } else {
                0.0
            }
        }),
    );

    Ok(LocalRbVarianceEstimate {
        estimate,
        diagnostics: LocalRbDiagnostics {
            deterministic_fraction,
            residual_variance_estimate: residual,
        },
    })
}

const DEFAULT_SELECTED_INVERSE_CLOSURE_MULTIPLIER: usize = 10;
const DEFAULT_EXACT_SOLVE_BLOCK_SIZE: usize = 4096;

#[derive(Debug, Clone)]
struct CholeskyColumn {
    diag: f64,
    descendants: Vec<(usize, f64)>,
}

#[derive(Debug)]
struct SelectedInverseWorkspace {
    inverse: SparseSelectedInverse,
    index: Vec<HashMap<usize, usize>>,
}

/// Compute selected inverse entries for a requested sparse pattern plus its
/// Takahashi dependency closure.
pub fn selected_inverse_entries(
    factor: &SparseCholeskyFactor,
    requested: &SparseSymmetricPattern,
) -> Result<SelectedInverseEntries, GmrfError> {
    selected_inverse_entries_with_limit(
        factor,
        requested,
        default_selected_inverse_closure_limit(factor),
    )
}

/// Compute selected inverse entries with an explicit Takahashi closure limit.
pub fn selected_inverse_entries_with_limit(
    factor: &SparseCholeskyFactor,
    requested: &SparseSymmetricPattern,
    closure_limit: usize,
) -> Result<SelectedInverseEntries, GmrfError> {
    if requested.dimension != factor.dimension() {
        return Err(GmrfError::DimensionMismatch(
            "selected inverse pattern dimension must match factor dimension",
        ));
    }
    let columns = cholesky_columns(factor)?;
    let factor_pairs = cholesky_factor_pair_count(&columns);
    let (closure, diagnostics) =
        takahashi_dependency_closure(&columns, requested, closure_limit, factor_pairs);
    let Some(closure) = closure else {
        return Ok(SelectedInverseEntries {
            inverse: None,
            diagnostics,
        });
    };

    let inverse = if closure_is_cholesky_factor_pattern(&columns, &closure, factor_pairs) {
        takahashi_selected_inverse_on_factor_pattern(&columns)?
    } else {
        takahashi_selected_inverse_on_closure(&columns, closure)?
    };
    Ok(SelectedInverseEntries {
        inverse: Some(inverse),
        diagnostics,
    })
}

/// Exact latent marginal variances from selected inverse diagonal.
pub fn selected_inverse_diag(factor: &SparseCholeskyFactor) -> Result<VarianceEstimate, GmrfError> {
    Ok(selected_inverse_diag_with_diagnostics(factor)?.estimate)
}

/// Exact latent marginal variances by repeated covariance solves.
pub fn exact_solve_diag(factor: &SparseCholeskyFactor) -> Result<VarianceEstimate, GmrfError> {
    exact_solve_diag_with_progress(factor, 0, |_, _| {})
}

/// Exact latent marginal variances by repeated covariance solves with progress callbacks.
pub fn exact_solve_diag_with_progress<F>(
    factor: &SparseCholeskyFactor,
    progress_interval: usize,
    progress: F,
) -> Result<VarianceEstimate, GmrfError>
where
    F: FnMut(usize, usize),
{
    exact_solve_diag_blocked_with_progress(
        factor,
        DEFAULT_EXACT_SOLVE_BLOCK_SIZE,
        progress_interval,
        progress,
    )
}

/// Exact latent marginal variances by blocked repeated covariance solves.
pub fn exact_solve_diag_blocked_with_progress<F>(
    factor: &SparseCholeskyFactor,
    block_size: usize,
    progress_interval: usize,
    mut progress: F,
) -> Result<VarianceEstimate, GmrfError>
where
    F: FnMut(usize, usize),
{
    if block_size == 0 {
        return Err(GmrfError::DimensionMismatch(
            "exact solve block size must be at least one",
        ));
    }
    let n = factor.dimension();
    let mut values = Vector::zeros(n);
    let mut next_progress = progress_interval;
    for start in (0..n).step_by(block_size) {
        let width = (n - start).min(block_size);
        let mut rhs = DenseMatrix::zeros(n, width);
        for local in 0..width {
            rhs[(start + local, local)] = 1.0;
        }
        factor.solve_dense_in_place(&mut rhs)?;
        for local in 0..width {
            values[start + local] = rhs[(start + local, local)];
        }
        let completed = start + width;
        if progress_interval > 0 && (completed == n || completed >= next_progress) {
            progress(completed, n);
            while next_progress <= completed {
                next_progress = next_progress.saturating_add(progress_interval);
            }
        }
    }
    finalize_variance_estimate(VarianceEstimator::ExactSolves, vec![values], vec![1])
}

/// Exact transformed variances by blocked covariance solves.
pub fn exact_solve_transformed_diag(
    factor: &SparseCholeskyFactor,
    operator: &SparseRowOperator,
) -> Result<VarianceEstimate, GmrfError> {
    exact_solve_transformed_diag_with_progress(
        factor,
        operator,
        DEFAULT_EXACT_SOLVE_BLOCK_SIZE,
        0,
        |_, _| {},
    )
}

/// Exact transformed variances by blocked covariance solves with progress callbacks.
pub fn exact_solve_transformed_diag_with_progress<F>(
    factor: &SparseCholeskyFactor,
    operator: &SparseRowOperator,
    block_size: usize,
    progress_interval: usize,
    mut progress: F,
) -> Result<VarianceEstimate, GmrfError>
where
    F: FnMut(usize, usize),
{
    if operator.ncols != factor.dimension() {
        return Err(GmrfError::DimensionMismatch(
            "operator columns must match factor dimension",
        ));
    }
    if block_size == 0 {
        return Err(GmrfError::DimensionMismatch(
            "exact solve block size must be at least one",
        ));
    }
    let output_dim = operator.nrows();
    let state_dim = factor.dimension();
    let mut values = Vector::zeros(output_dim);
    let mut next_progress = progress_interval;
    for start in (0..output_dim).step_by(block_size) {
        let width = (output_dim - start).min(block_size);
        let mut rhs = DenseMatrix::zeros(state_dim, width);
        for local in 0..width {
            for &(state, weight) in &operator.rows[start + local] {
                rhs[(state, local)] = weight;
            }
        }
        factor.solve_dense_in_place(&mut rhs)?;
        for local in 0..width {
            let row = &operator.rows[start + local];
            let variance = row
                .iter()
                .map(|(state, weight)| *weight * rhs[(*state, local)])
                .sum::<f64>();
            values[start + local] = variance.max(0.0);
        }
        let completed = start + width;
        if progress_interval > 0 && (completed == output_dim || completed >= next_progress) {
            progress(completed, output_dim);
            while next_progress <= completed {
                next_progress = next_progress.saturating_add(progress_interval);
            }
        }
    }
    finalize_variance_estimate(VarianceEstimator::ExactSolves, vec![values], vec![1])
}

/// Exact latent marginal variances from selected inverse diagonal with closure diagnostics.
pub fn selected_inverse_diag_with_diagnostics(
    factor: &SparseCholeskyFactor,
) -> Result<SelectedInverseDiagResult, GmrfError> {
    let n = factor.dimension();
    let mut requested = SparseSymmetricPattern::new(n);
    for index in 0..n {
        requested.insert(index, index)?;
    }
    let selected = selected_inverse_entries(factor, &requested)?;
    let inverse = selected
        .inverse
        .ok_or(GmrfError::SelectedInversePatternNotCovered)?;
    let diag_tilde = inverse.diag();
    let values = factor.permute_factor_to_original(&diag_tilde)?;
    let estimate =
        finalize_variance_estimate(VarianceEstimator::SelectedInverse, vec![values], vec![1])?;
    Ok(SelectedInverseDiagResult {
        estimate,
        diagnostics: selected.diagnostics,
    })
}

/// Exact transformed variances from selected inverse when the Takahashi closure fits the limit.
pub fn selected_inverse_transformed_diag(
    factor: &SparseCholeskyFactor,
    operator: &SparseRowOperator,
) -> Result<SelectedInverseTransformedResult, GmrfError> {
    if operator.ncols != factor.dimension() {
        return Err(GmrfError::DimensionMismatch(
            "operator columns must match factor dimension",
        ));
    }
    let operator_tilde = operator.permute_columns_to_factor(&factor.permutation())?;
    let requested = SparseSymmetricPattern::from_transformed_operator(&operator_tilde)?;
    let selected = selected_inverse_entries(factor, &requested)?;
    let Some(inverse) = selected.inverse else {
        return Ok(SelectedInverseTransformedResult {
            estimate: None,
            diagnostics: selected.diagnostics,
        });
    };

    let mut values = Vector::zeros(operator.nrows());
    for (row_index, row) in operator_tilde.rows.iter().enumerate() {
        let mut value = 0.0;
        for (col_a, weight_a) in row {
            for (col_b, weight_b) in row {
                let entry = inverse
                    .get(*col_a, *col_b)
                    .ok_or(GmrfError::SelectedInversePatternNotCovered)?;
                value += *weight_a * *weight_b * entry;
            }
        }
        values[row_index] = value;
    }
    let estimate =
        finalize_variance_estimate(VarianceEstimator::SelectedInverse, vec![values], vec![1])?;

    Ok(SelectedInverseTransformedResult {
        estimate: Some(estimate),
        diagnostics: selected.diagnostics,
    })
}

fn default_selected_inverse_closure_limit(factor: &SparseCholeskyFactor) -> usize {
    DEFAULT_SELECTED_INVERSE_CLOSURE_MULTIPLIER.saturating_mul(factor.nnz())
}

fn cholesky_columns(factor: &SparseCholeskyFactor) -> Result<Vec<CholeskyColumn>, GmrfError> {
    let n = factor.dimension();
    let mut raw_columns = vec![Vec::<(usize, f64)>::new(); n];
    for (row, col, value) in factor.lower_triplets() {
        raw_columns[col].push((row, value));
    }
    let mut columns = Vec::with_capacity(n);
    for (col, column) in raw_columns.iter_mut().enumerate() {
        column.sort_by_key(|(row, _)| *row);
        let diag = column
            .iter()
            .find_map(|(row, value)| (*row == col).then_some(*value))
            .ok_or(GmrfError::NonPositiveDefinite)?;
        if diag <= 0.0 {
            return Err(GmrfError::NonPositiveDefinite);
        }
        let descendants = column
            .iter()
            .copied()
            .filter(|(row, _)| *row > col)
            .collect::<Vec<_>>();
        columns.push(CholeskyColumn { diag, descendants });
    }
    Ok(columns)
}

fn cholesky_factor_pair_count(columns: &[CholeskyColumn]) -> usize {
    columns
        .iter()
        .map(|column| 1 + column.descendants.len())
        .sum()
}

fn closure_is_cholesky_factor_pattern(
    columns: &[CholeskyColumn],
    closure: &[BTreeSet<usize>],
    factor_pairs: usize,
) -> bool {
    if closure.len() != columns.len() {
        return false;
    }
    if closure.iter().map(BTreeSet::len).sum::<usize>() != factor_pairs {
        return false;
    }
    columns.iter().enumerate().all(|(col, column)| {
        let rows = &closure[col];
        rows.contains(&col)
            && rows.len() == 1 + column.descendants.len()
            && column.descendants.iter().all(|(row, _)| rows.contains(row))
    })
}

fn takahashi_dependency_closure(
    columns: &[CholeskyColumn],
    requested: &SparseSymmetricPattern,
    closure_limit: usize,
    factor_pairs: usize,
) -> (Option<Vec<BTreeSet<usize>>>, SelectedInverseDiagnostics) {
    let n = columns.len();
    let mut closure = vec![BTreeSet::<usize>::new(); n];
    let mut queue = VecDeque::<(usize, usize)>::new();
    let mut closure_pairs = 0usize;

    for &(row, col) in &requested.pairs {
        if insert_closure_pair(&mut closure, row, col, &mut closure_pairs) {
            queue.push_back(normalize_pair(row, col));
            if closure_pairs > closure_limit {
                return (
                    None,
                    selected_inverse_diagnostics(
                        requested.pairs.len(),
                        closure_pairs,
                        factor_pairs,
                        closure_limit,
                        SelectedInverseStatus::ClosureTooLarge,
                    ),
                );
            }
        }
    }

    while let Some((row, col)) = queue.pop_front() {
        for &(descendant, _) in &columns[col].descendants {
            let dependency = normalize_pair(row, descendant);
            if insert_closure_pair(&mut closure, dependency.0, dependency.1, &mut closure_pairs) {
                queue.push_back(dependency);
                if closure_pairs > closure_limit {
                    return (
                        None,
                        selected_inverse_diagnostics(
                            requested.pairs.len(),
                            closure_pairs,
                            factor_pairs,
                            closure_limit,
                            SelectedInverseStatus::ClosureTooLarge,
                        ),
                    );
                }
            }
        }
    }

    (
        Some(closure),
        selected_inverse_diagnostics(
            requested.pairs.len(),
            closure_pairs,
            factor_pairs,
            closure_limit,
            SelectedInverseStatus::Complete,
        ),
    )
}

fn insert_closure_pair(
    closure: &mut [BTreeSet<usize>],
    row: usize,
    col: usize,
    closure_pairs: &mut usize,
) -> bool {
    let (row, col) = normalize_pair(row, col);
    let inserted = closure[col].insert(row);
    if inserted {
        *closure_pairs += 1;
    }
    inserted
}

fn selected_inverse_diagnostics(
    requested_pairs: usize,
    closure_pairs: usize,
    factor_pairs: usize,
    closure_limit: usize,
    status: SelectedInverseStatus,
) -> SelectedInverseDiagnostics {
    SelectedInverseDiagnostics {
        requested_pairs,
        closure_pairs,
        factor_pairs,
        closure_over_factor: closure_pairs as f64 / factor_pairs.max(1) as f64,
        closure_limit,
        status,
    }
}

fn takahashi_selected_inverse_on_closure(
    columns: &[CholeskyColumn],
    closure: Vec<BTreeSet<usize>>,
) -> Result<SparseSelectedInverse, GmrfError> {
    let mut workspace = selected_inverse_workspace(closure);
    let n = columns.len();
    for j in (0..n).rev() {
        let diag = columns[j].diag;
        let selected_rows = workspace.inverse.columns[j].rows.clone();
        for i in selected_rows.iter().copied().filter(|row| *row > j) {
            let sum = columns[j]
                .descendants
                .iter()
                .try_fold(0.0, |sum, (k, l_kj)| {
                    selected_inverse_workspace_get(&workspace, i, *k)
                        .map(|value| sum + *l_kj * value)
                })?;
            selected_inverse_workspace_set(&mut workspace, i, j, -sum / diag)?;
        }

        if selected_inverse_workspace_contains(&workspace, j, j) {
            let diag_sum = columns[j]
                .descendants
                .iter()
                .try_fold(0.0, |sum, (k, l_kj)| {
                    selected_inverse_workspace_get(&workspace, *k, j)
                        .map(|value| sum + *l_kj * value)
                })?;
            selected_inverse_workspace_set(
                &mut workspace,
                j,
                j,
                1.0 / (diag * diag) - diag_sum / diag,
            )?;
        }
    }

    Ok(workspace.inverse)
}

fn takahashi_selected_inverse_on_factor_pattern(
    columns: &[CholeskyColumn],
) -> Result<SparseSelectedInverse, GmrfError> {
    let n = columns.len();
    let mut inverse = selected_inverse_factor_pattern_storage(columns);
    let marker_empty = usize::MAX;
    let mut marker = vec![marker_empty; n];
    let mut weights = Vec::<f64>::new();
    let mut sums = Vec::<f64>::new();

    for j in (0..n).rev() {
        let descendants = &columns[j].descendants;
        let degree = descendants.len();
        weights.clear();
        weights.resize(degree, 0.0);
        sums.clear();
        sums.resize(degree, 0.0);

        for (position, (row, value)) in descendants.iter().copied().enumerate() {
            marker[row] = position;
            weights[position] = value;
        }

        for (column_position, (column, column_weight)) in descendants.iter().copied().enumerate() {
            let inverse_column = &inverse.columns[column];
            for (row, value) in inverse_column
                .rows
                .iter()
                .copied()
                .zip(inverse_column.values.iter().copied())
            {
                let row_position = marker[row];
                if row_position == marker_empty {
                    continue;
                }
                sums[row_position] += value * column_weight;
                if row_position != column_position {
                    sums[column_position] += value * weights[row_position];
                }
            }
        }

        let diagonal = columns[j].diag;
        let current_values = &mut inverse.columns[j].values;
        let mut diagonal_sum = 0.0;
        for (position, value) in sums.iter().copied().enumerate() {
            let inverse_value = -value / diagonal;
            current_values[position + 1] = inverse_value;
            diagonal_sum += weights[position] * inverse_value;
        }
        current_values[0] = 1.0 / (diagonal * diagonal) - diagonal_sum / diagonal;

        for (row, _) in descendants {
            marker[*row] = marker_empty;
        }
    }

    Ok(inverse)
}

fn selected_inverse_factor_pattern_storage(columns: &[CholeskyColumn]) -> SparseSelectedInverse {
    let inverse_columns = columns
        .iter()
        .enumerate()
        .map(|(col, column)| {
            let mut rows = Vec::with_capacity(1 + column.descendants.len());
            rows.push(col);
            rows.extend(column.descendants.iter().map(|(row, _)| *row));
            let values = vec![0.0; rows.len()];
            SparseSelectedInverseColumn { rows, values }
        })
        .collect::<Vec<_>>();
    SparseSelectedInverse {
        dimension: columns.len(),
        columns: inverse_columns,
    }
}

fn selected_inverse_workspace(closure: Vec<BTreeSet<usize>>) -> SelectedInverseWorkspace {
    let dimension = closure.len();
    let mut columns = Vec::with_capacity(dimension);
    let mut index = Vec::with_capacity(dimension);
    for rows in closure {
        let rows = rows.into_iter().collect::<Vec<_>>();
        let values = vec![0.0; rows.len()];
        let column_index = rows
            .iter()
            .copied()
            .enumerate()
            .map(|(index, row)| (row, index))
            .collect::<HashMap<_, _>>();
        columns.push(SparseSelectedInverseColumn { rows, values });
        index.push(column_index);
    }
    SelectedInverseWorkspace {
        inverse: SparseSelectedInverse { dimension, columns },
        index,
    }
}

fn selected_inverse_workspace_contains(
    workspace: &SelectedInverseWorkspace,
    row: usize,
    col: usize,
) -> bool {
    let (row, col) = normalize_pair(row, col);
    workspace.index[col].contains_key(&row)
}

fn selected_inverse_workspace_get(
    workspace: &SelectedInverseWorkspace,
    row: usize,
    col: usize,
) -> Result<f64, GmrfError> {
    let (row, col) = normalize_pair(row, col);
    let index = workspace.index[col]
        .get(&row)
        .copied()
        .ok_or(GmrfError::SelectedInversePatternNotCovered)?;
    Ok(workspace.inverse.columns[col].values[index])
}

fn selected_inverse_workspace_set(
    workspace: &mut SelectedInverseWorkspace,
    row: usize,
    col: usize,
    value: f64,
) -> Result<(), GmrfError> {
    let (row, col) = normalize_pair(row, col);
    let index = workspace.index[col]
        .get(&row)
        .copied()
        .ok_or(GmrfError::SelectedInversePatternNotCovered)?;
    workspace.inverse.columns[col].values[index] = value;
    Ok(())
}

/// One Hutchinson batch for transformed variances using a custom covariance action.
pub fn transformed_hutchinson_variances_batch_with_solve<R, F>(
    operator: &SparseRowOperator,
    num_samples: usize,
    rng: &mut R,
    mut solve_covariance_action: F,
) -> Result<Vector, GmrfError>
where
    R: Rng + ?Sized,
    F: FnMut(&Vector) -> Result<Vector, GmrfError>,
{
    if num_samples == 0 {
        return Err(GmrfError::DimensionMismatch(
            "at least one Hutchinson probe is required",
        ));
    }

    let output_dim = operator.nrows();
    let mut variances = Vector::zeros(output_dim);
    for _ in 0..num_samples {
        let probe = draw_probe(output_dim, ProbeDistribution::Rademacher, rng);
        let rhs = operator.apply_transpose(&probe)?;
        let solved = solve_covariance_action(&rhs)?;
        let projected = operator.apply(&solved)?;
        variances += projected.component_mul(&probe);
    }

    Ok(variances / (num_samples as f64))
}

/// Estimate transformed variances with deterministic batched Hutchinson probes and a
/// custom covariance action.
pub fn estimate_batched_transformed_hutchinson_with_solve<F>(
    operator: &SparseRowOperator,
    config: ProbeBatchConfig,
    floor: VarianceFloor,
    mut solve_covariance_action: F,
) -> Result<BatchedVarianceEstimate, GmrfError>
where
    F: FnMut(&Vector) -> Result<Vector, GmrfError>,
{
    let batch_sizes = probe_batch_sizes(config.num_probes, config.batch_count)?;
    let mut batch_estimates = Vec::with_capacity(batch_sizes.len());
    let mut floor_hits = 0_usize;

    for (batch_index, batch_size) in batch_sizes.iter().copied().enumerate() {
        let mut rng =
            rand::rngs::StdRng::seed_from_u64(probe_batch_seed(config.rng_seed, batch_index));
        let raw = transformed_hutchinson_variances_batch_with_solve(
            operator,
            batch_size,
            &mut rng,
            |rhs| solve_covariance_action(rhs),
        )?;
        let stabilized = stabilize_variances(&raw, floor)?;
        floor_hits += stabilized.floor_hits;
        batch_estimates.push(stabilized.values);
    }

    let estimate = weighted_average_vectors(&batch_estimates, &batch_sizes)?;
    Ok(BatchedVarianceEstimate {
        estimate,
        batch_estimates,
        batch_sizes,
        floor_hits,
    })
}

/// Estimate transformed variance decomposition with deterministic batched Hutchinson probes.
pub fn estimate_batched_transformed_hutchinson_decomposition(
    gmrf: &mut Gmrf,
    operator: &SparseRowOperator,
    constraints: &DenseMatrix,
    config: ProbeBatchConfig,
    floor: VarianceFloor,
) -> Result<BatchedTransformedVarianceDecomposition, GmrfError> {
    let unconstrained =
        estimate_batched_transformed_hutchinson_with_solve(operator, config, floor, |rhs| {
            gmrf.solve_precision(rhs)
        })?;
    let removed_diag = gmrf.transformed_variance_correction_diag(operator, constraints)?;
    let constrained = stabilize_removed_variances(&unconstrained.estimate, &removed_diag, floor)?;

    Ok(BatchedTransformedVarianceDecomposition {
        decomposition: TransformedVarianceDecomposition {
            unconstrained_diag: unconstrained.estimate,
            constrained_diag: constrained.values,
            removed_diag,
        },
        batch_estimates: unconstrained.batch_estimates,
        batch_sizes: unconstrained.batch_sizes,
        unconstrained_floor_hits: unconstrained.floor_hits,
        constrained_floor_hits: constrained.floor_hits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CooMatrix, SparseMatrix};
    use crate::SparseRowOperator;

    fn identity_precision(size: usize) -> SparseMatrix {
        let mut coo = CooMatrix::new(size, size);
        for i in 0..size {
            coo.push(i, i, 1.0);
        }
        SparseMatrix::from(&coo)
    }

    fn tridiagonal_precision() -> SparseMatrix {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 4.0);
        coo.push(0, 1, 1.0);
        coo.push(1, 0, 1.0);
        coo.push(1, 1, 3.0);
        coo.push(1, 2, 0.5);
        coo.push(2, 1, 0.5);
        coo.push(2, 2, 2.0);
        SparseMatrix::from(&coo)
    }

    fn permuted_four_node_precision() -> SparseMatrix {
        let permutation = [2usize, 0, 3, 1];
        let entries = [
            (0usize, 0usize, 5.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 4.0),
            (1, 2, -0.75),
            (2, 1, -0.75),
            (2, 2, 3.5),
            (2, 3, -0.5),
            (3, 2, -0.5),
            (3, 3, 2.5),
        ];
        let mut coo = CooMatrix::new(4, 4);
        for (row, col, value) in entries {
            coo.push(permutation[row], permutation[col], value);
        }
        SparseMatrix::from(&coo)
    }

    fn repeated_solve_diag(precision: &SparseMatrix) -> Vector {
        let factor = precision.cholesky_sqrt_lower().unwrap();
        Vector::from_fn(precision.nrows(), |i| {
            let mut rhs = Vector::zeros(precision.nrows());
            rhs[i] = 1.0;
            factor.solve_in_place(&mut rhs).unwrap();
            rhs[i]
        })
    }

    fn repeated_solve_transformed_diag(
        precision: &SparseMatrix,
        operator: &SparseRowOperator,
    ) -> Vector {
        let factor = precision.cholesky_sqrt_lower().unwrap();
        Vector::from_fn(operator.nrows(), |row_index| {
            let rhs = operator.row_as_vector(row_index).unwrap();
            let solved = factor.solve(&rhs).unwrap();
            rhs.dot(&solved)
        })
    }

    fn dense_covariance_from_repeated_solves(precision: &SparseMatrix) -> Vec<Vec<f64>> {
        let factor = precision.cholesky_sqrt_lower().unwrap();
        (0..precision.ncols())
            .map(|col| {
                let mut rhs = Vector::zeros(precision.ncols());
                rhs[col] = 1.0;
                factor.solve(&rhs).unwrap().as_slice().to_vec()
            })
            .collect()
    }

    fn factor_pattern_closure_for_test(columns: &[CholeskyColumn]) -> Vec<BTreeSet<usize>> {
        columns
            .iter()
            .enumerate()
            .map(|(col, column)| {
                let mut rows = BTreeSet::new();
                rows.insert(col);
                rows.extend(column.descendants.iter().map(|(row, _)| *row));
                rows
            })
            .collect()
    }

    #[test]
    fn probe_batch_sizes_split_evenly() {
        assert_eq!(probe_batch_sizes(10, 3).unwrap(), vec![4, 3, 3]);
        assert_eq!(probe_batch_sizes(2, 8).unwrap(), vec![1, 1]);
        assert!(probe_batch_sizes(0, 1).is_err());
        assert!(probe_batch_sizes(1, 0).is_err());
    }

    #[test]
    fn weighted_average_vectors_respects_weights() {
        let lhs = Vector::from_vec(vec![1.0, 3.0]);
        let rhs = Vector::from_vec(vec![3.0, 7.0]);
        let avg = weighted_average_vectors(&[lhs, rhs], &[1, 3]).unwrap();
        assert!((avg[0] - 2.5).abs() < 1e-12);
        assert!((avg[1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn positive_mean_floor_reports_hits() {
        let values = Vector::from_vec(vec![2.0, -1e-14, 0.0]);
        let stabilized =
            stabilize_variances(&values, VarianceFloor::PositiveMean { scale: 1e-12 }).unwrap();
        assert_eq!(stabilized.floor_hits, 2);
        assert!(stabilized.values[1] > 0.0);
        assert!(stabilized.values[2] > 0.0);
    }

    #[test]
    fn batched_transformed_hutchinson_runs_for_identity() {
        let precision = identity_precision(2);
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let mut gmrf = Gmrf::from_mean_and_precision(Vector::zeros(2), precision)
            .unwrap()
            .with_precision_sqrt(factor);
        let constraints = DenseMatrix::zeros(0, 2);
        let operator = SparseRowOperator::identity(2);
        let estimate = estimate_batched_transformed_hutchinson_decomposition(
            &mut gmrf,
            &operator,
            &constraints,
            ProbeBatchConfig {
                num_probes: 8,
                batch_count: 4,
                rng_seed: 3,
            },
            VarianceFloor::Zero,
        )
        .unwrap();
        assert_eq!(estimate.decomposition.unconstrained_diag.len(), 2);
        assert_eq!(estimate.batch_sizes, vec![2, 2, 2, 2]);
    }

    #[test]
    fn weighted_trace_exact_sparse_matches_dense_covariance() {
        let precision = tridiagonal_precision();
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let mut weight = CooMatrix::new(3, 3);
        weight.push(0, 0, 2.0);
        weight.push(0, 1, 0.25);
        weight.push(1, 0, 0.25);
        weight.push(1, 1, 3.0);
        weight.push(2, 2, 0.5);
        let weight = SparseMatrix::from(&weight);

        let estimate = exact_weighted_covariance_trace(&factor, &weight).unwrap();
        let covariance_columns = dense_covariance_from_repeated_solves(&precision);
        let reference = weight
            .triplet_iter()
            .map(|(row, col, value)| *value * covariance_columns[col][row])
            .sum::<f64>();

        assert_eq!(estimate.estimator, VarianceEstimator::ExactSolves);
        assert!((estimate.value - reference).abs() < 1e-10);
    }

    #[test]
    fn weighted_trace_transformed_exact_matches_dense_covariance() {
        let precision = tridiagonal_precision();
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let operator =
            SparseRowOperator::new(3, vec![vec![(0, 1.0), (1, -1.0)], vec![(1, 0.5), (2, 2.0)]])
                .unwrap();
        let weights = Vector::from_vec(vec![2.0, 0.25]);

        let estimate =
            exact_weighted_transformed_covariance_trace(&factor, &operator, &weights).unwrap();
        let variances = repeated_solve_transformed_diag(&precision, &operator);
        let reference = variances.dot(&weights);

        assert_eq!(estimate.estimator, VarianceEstimator::ExactSolves);
        assert!((estimate.value - reference).abs() < 1e-10);
    }

    #[test]
    fn weighted_trace_hutchinson_is_exact_for_diagonal_identity() {
        let precision = identity_precision(3);
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let mut weight = CooMatrix::new(3, 3);
        weight.push(0, 0, 2.0);
        weight.push(1, 1, 3.0);
        weight.push(2, 2, 5.0);
        let weight = SparseMatrix::from(&weight);

        let estimate = estimate_hutchinson_weighted_covariance_trace(
            &factor,
            &weight,
            ProbeBatchConfig {
                num_probes: 4,
                batch_count: 2,
                rng_seed: 11,
            },
            ProbeDistribution::Rademacher,
        )
        .unwrap();

        assert_eq!(estimate.estimator, VarianceEstimator::Hutchinson);
        assert_eq!(estimate.batch_sizes, vec![2, 2]);
        assert!((estimate.value - 10.0).abs() < 1e-12);
    }

    #[test]
    fn weighted_trace_transformed_hutchinson_is_exact_for_identity_rows() {
        let precision = identity_precision(2);
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let operator = SparseRowOperator::identity(2);
        let weights = Vector::from_vec(vec![1.5, 2.5]);

        let estimate = estimate_hutchinson_weighted_transformed_covariance_trace(
            &factor,
            &operator,
            &weights,
            ProbeBatchConfig {
                num_probes: 4,
                batch_count: 2,
                rng_seed: 13,
            },
            ProbeDistribution::Rademacher,
        )
        .unwrap();

        assert_eq!(estimate.estimator, VarianceEstimator::Hutchinson);
        assert!((estimate.value - 4.0).abs() < 1e-12);
    }

    #[test]
    fn paired_exact_weighted_trace_matches_separate_exact_path() {
        let precision = tridiagonal_precision();
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let operator = SparseRowOperator::new(
            3,
            vec![vec![(0, 1.0), (1, -1.0)], vec![(1, 1.0), (2, -1.0)]],
        )
        .expect("valid operator");
        let weights = Vector::from_vec(vec![0.75, 2.0]);

        let paired =
            exact_transformed_variance_weighted_trace(&factor, &operator, &weights).unwrap();
        let exact_diag = exact_solve_transformed_diag(&factor, &operator).unwrap();
        let exact_trace =
            exact_weighted_transformed_covariance_trace(&factor, &operator, &weights).unwrap();

        assert_eq!(paired.variances.estimator, VarianceEstimator::ExactSolves);
        assert_eq!(
            paired.weighted_trace.estimator,
            VarianceEstimator::ExactSolves
        );
        assert!((&paired.variances.values - &exact_diag.values).norm() < 1e-12);
        assert!((paired.weighted_trace.value - exact_trace.value).abs() < 1e-12);
        assert!(
            (paired.weighted_trace.value - paired.variances.values.dot(&weights)).abs() < 1e-12
        );
    }

    #[test]
    fn paired_hutchinson_weighted_trace_is_dot_of_returned_batches() {
        let precision = identity_precision(3);
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let operator = SparseRowOperator::identity(3);
        let weights = Vector::from_vec(vec![0.5, 1.25, 2.0]);

        let paired = estimate_hutchinson_transformed_variance_weighted_trace(
            &factor,
            &operator,
            &weights,
            ProbeBatchConfig {
                num_probes: 6,
                batch_count: 3,
                rng_seed: 19,
            },
            VarianceFloor::Zero,
            ProbeDistribution::Rademacher,
        )
        .unwrap();

        let total = paired.variances.batch_sizes.iter().sum::<usize>() as f64;
        let trace_from_variance_batches = paired
            .variance_batch_estimates
            .iter()
            .zip(paired.variances.batch_sizes.iter().copied())
            .map(|(batch, size)| batch.dot(&weights) * size as f64 / total)
            .sum::<f64>();

        assert_eq!(
            paired.weighted_trace.estimator,
            VarianceEstimator::Hutchinson
        );
        assert!((paired.weighted_trace.value - trace_from_variance_batches).abs() < 1e-12);
        assert!(
            (paired.weighted_trace.value - paired.variances.values.dot(&weights)).abs() < 1e-12
        );
    }

    #[test]
    fn paired_hutchinson_weighted_trace_solves_once_per_probe() {
        let precision = identity_precision(2);
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let operator = SparseRowOperator::identity(2);
        let weights = Vector::from_vec(vec![1.0, 1.0]);
        let mut solve_count = 0_usize;

        let paired = estimate_hutchinson_transformed_variance_weighted_trace_with_solve(
            &operator,
            &weights,
            ProbeBatchConfig {
                num_probes: 7,
                batch_count: 3,
                rng_seed: 23,
            },
            VarianceFloor::Zero,
            ProbeDistribution::Rademacher,
            |rhs| {
                solve_count += 1;
                factor.solve(rhs)
            },
        )
        .unwrap();

        assert_eq!(solve_count, 7);
        assert!((paired.weighted_trace.value - 2.0).abs() < 1e-12);
    }

    #[test]
    fn factored_transformed_variance_dispatch_selects_exact_and_hutchinson() {
        let precision = identity_precision(3);
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let operator = SparseRowOperator::identity(3);
        let config = ProbeBatchConfig {
            num_probes: 4,
            batch_count: 2,
            rng_seed: 31,
        };

        let exact = estimate_factored_transformed_variances(
            &factor,
            &operator,
            TransformedVarianceMode::Exact,
        )
        .unwrap();
        let auto_exact = estimate_factored_transformed_variances(
            &factor,
            &operator,
            TransformedVarianceMode::Auto {
                exact_max_dofs: 3,
                config,
                floor: VarianceFloor::Zero,
                distribution: ProbeDistribution::Rademacher,
            },
        )
        .unwrap();
        let hutchinson = estimate_factored_transformed_variances(
            &factor,
            &operator,
            TransformedVarianceMode::Auto {
                exact_max_dofs: 2,
                config,
                floor: VarianceFloor::Zero,
                distribution: ProbeDistribution::Rademacher,
            },
        )
        .unwrap();

        assert_eq!(exact.estimator, VarianceEstimator::ExactSolves);
        assert_eq!(auto_exact.estimator, VarianceEstimator::ExactSolves);
        assert_eq!(hutchinson.estimator, VarianceEstimator::Hutchinson);
        for estimate in [&exact, &auto_exact, &hutchinson] {
            assert!((&estimate.values - Vector::from_element(3, 1.0)).norm() < 1e-12);
        }
    }

    #[test]
    fn factored_weighted_trace_dispatch_reuses_variance_batches() {
        let precision = identity_precision(2);
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let operator = SparseRowOperator::identity(2);
        let weights = Vector::from_vec(vec![0.25, 1.75]);
        let mode = TransformedVarianceMode::Hutchinson {
            config: ProbeBatchConfig {
                num_probes: 6,
                batch_count: 3,
                rng_seed: 37,
            },
            floor: VarianceFloor::Zero,
            distribution: ProbeDistribution::Rademacher,
        };

        let estimate = estimate_factored_transformed_variance_weighted_trace(
            &factor, &operator, &weights, mode,
        )
        .unwrap();
        let total = estimate.variances.batch_sizes.iter().sum::<usize>() as f64;
        let trace_from_batches = estimate
            .variance_batch_estimates
            .iter()
            .zip(estimate.variances.batch_sizes.iter().copied())
            .map(|(batch, size)| batch.dot(&weights) * size as f64 / total)
            .sum::<f64>();

        assert_eq!(estimate.variances.estimator, VarianceEstimator::Hutchinson);
        assert_eq!(
            estimate.weighted_trace.estimator,
            VarianceEstimator::Hutchinson
        );
        assert!((estimate.weighted_trace.value - 2.0).abs() < 1e-12);
        assert!((estimate.weighted_trace.value - trace_from_batches).abs() < 1e-12);
    }

    #[test]
    fn constrained_transformed_variance_dispatch_uses_solver() {
        let precision = identity_precision(2);
        let constraints = DenseMatrix::from_fn(1, 2, |_, _| 1.0);
        let solver = ConstrainedPrecisionSolver::new(&precision, &constraints).unwrap();
        let operator = SparseRowOperator::new(2, vec![vec![(0, 1.0), (1, -1.0)]]).unwrap();
        let config = ProbeBatchConfig {
            num_probes: 4,
            batch_count: 2,
            rng_seed: 41,
        };

        let exact = estimate_constrained_transformed_variances(
            &solver,
            &operator,
            TransformedVarianceMode::Exact,
        )
        .unwrap();
        let hutchinson = estimate_constrained_transformed_variances(
            &solver,
            &operator,
            TransformedVarianceMode::Hutchinson {
                config,
                floor: VarianceFloor::Zero,
                distribution: ProbeDistribution::Rademacher,
            },
        )
        .unwrap();

        assert_eq!(exact.estimator, VarianceEstimator::ExactSolves);
        assert_eq!(hutchinson.estimator, VarianceEstimator::Hutchinson);
        assert!((exact.values[0] - 2.0).abs() < 1e-12);
        assert!((hutchinson.values[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn constrained_mc_variances_runs() {
        let precision = identity_precision(2);
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let mut gmrf = Gmrf::from_mean_and_precision(Vector::zeros(2), precision)
            .unwrap()
            .with_precision_sqrt(factor);
        let constraints = DenseMatrix::from_fn(1, 2, |_, _| 1.0);
        let rhs = Vector::zeros(1);
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let variances =
            estimate_constrained_mc_variances(&mut gmrf, &constraints, &rhs, 4, &mut rng).unwrap();
        assert_eq!(variances.len(), 2);
    }

    #[test]
    fn batched_constrained_mc_is_seeded_and_supports_transforms() {
        let make_gmrf = || {
            let precision = identity_precision(2);
            let factor = precision.cholesky_sqrt_lower().unwrap();
            Gmrf::from_mean_and_precision(Vector::zeros(2), precision)
                .unwrap()
                .with_precision_sqrt(factor)
        };
        let constraints = DenseMatrix::from_fn(1, 2, |_, _| 1.0);
        let rhs = Vector::zeros(1);
        let mut first_gmrf = make_gmrf();
        let mut second_gmrf = make_gmrf();
        let first = estimate_monte_carlo_constrained_variances(
            &mut first_gmrf,
            &constraints,
            &rhs,
            4096,
            8,
            17,
        )
        .unwrap();
        let second = estimate_monte_carlo_constrained_variances(
            &mut second_gmrf,
            &constraints,
            &rhs,
            4096,
            8,
            17,
        )
        .unwrap();
        assert_eq!(first.values, second.values);
        assert_eq!(first.batch_sizes, vec![512; 8]);
        assert!((first.values[0] - 0.5).abs() < 0.05);
        assert!((first.values[1] - 0.5).abs() < 0.05);

        let difference = SparseRowOperator::new(2, vec![vec![(0, 1.0), (1, -1.0)]]).unwrap();
        let mut transformed_gmrf = make_gmrf();
        let transformed = estimate_monte_carlo_constrained_transformed_variances(
            &mut transformed_gmrf,
            &difference,
            &constraints,
            &rhs,
            4096,
            8,
            17,
        )
        .unwrap();
        assert!((transformed.values[0] - 2.0).abs() < 0.15);
        assert!(transformed.batch_standard_error.is_some());
    }

    #[test]
    fn selected_inverse_diag_matches_repeated_solves() {
        let precision = tridiagonal_precision();
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let selected = selected_inverse_diag_with_diagnostics(&factor).unwrap();
        let reference = repeated_solve_diag(&precision);
        assert!((&selected.estimate.values - &reference).norm() < 1e-10);
        assert_eq!(selected.diagnostics.status, SelectedInverseStatus::Complete);
    }

    #[test]
    fn exact_solve_diag_matches_repeated_solves_for_permuted_input() {
        let precision = permuted_four_node_precision();
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let exact = exact_solve_diag(&factor).unwrap();
        let reference = repeated_solve_diag(&precision);
        assert_eq!(exact.estimator, VarianceEstimator::ExactSolves);
        assert!((&exact.values - &reference).norm() < 1e-10);
    }

    #[test]
    fn exact_solve_transformed_diag_matches_repeated_solves() {
        let precision = tridiagonal_precision();
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let operator = SparseRowOperator::new(
            3,
            vec![vec![(0, 1.0), (1, -1.0)], vec![(1, 1.0), (2, -1.0)]],
        )
        .expect("valid operator");
        let exact = exact_solve_transformed_diag(&factor, &operator).unwrap();
        let reference = repeated_solve_transformed_diag(&precision, &operator);
        assert_eq!(exact.estimator, VarianceEstimator::ExactSolves);
        assert!((&exact.values - &reference).norm() < 1e-10);
    }

    #[test]
    fn selected_inverse_diag_matches_repeated_solves_for_permuted_input() {
        let precision = permuted_four_node_precision();
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let selected = selected_inverse_diag(&factor).unwrap();
        let reference = repeated_solve_diag(&precision);
        assert!((&selected.values - &reference).norm() < 1e-10);
    }

    #[test]
    fn selected_inverse_factor_fast_path_matches_generic_closure() {
        let precision = permuted_four_node_precision();
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let columns = cholesky_columns(&factor).unwrap();
        let generic = takahashi_selected_inverse_on_closure(
            &columns,
            factor_pattern_closure_for_test(&columns),
        )
        .unwrap();
        let fast = takahashi_selected_inverse_on_factor_pattern(&columns).unwrap();

        for col in 0..columns.len() {
            assert_eq!(fast.columns[col].rows, generic.columns[col].rows);
            for row in fast.columns[col].rows.iter().copied() {
                let fast_value = fast.get(row, col).unwrap();
                let generic_value = generic.get(row, col).unwrap();
                assert!((fast_value - generic_value).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn hutchinson_identity_is_exact_with_rademacher_probe() {
        let precision = identity_precision(4);
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let mut gmrf = Gmrf::from_mean_and_precision(Vector::zeros(4), precision)
            .unwrap()
            .with_precision_sqrt(factor);
        let estimate =
            estimate_hutchinson_variances(&mut gmrf, 1, 1, 99, ProbeDistribution::Rademacher)
                .unwrap();
        for value in estimate.values.iter() {
            assert!((*value - 1.0).abs() < 1e-12);
        }
        assert_eq!(estimate.num_negative, 0);
    }

    #[test]
    fn monte_carlo_variances_runs_with_diagnostics() {
        let precision = identity_precision(2);
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let mut gmrf = Gmrf::from_mean_and_precision(Vector::zeros(2), precision)
            .unwrap()
            .with_precision_sqrt(factor);
        let estimate = estimate_monte_carlo_variances(&mut gmrf, 64, 4, 5).unwrap();
        assert_eq!(estimate.values.len(), 2);
        assert!(estimate.batch_standard_error.is_some());
        assert!(estimate.values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn local_rb_full_block_recovers_exact_latent_diag() {
        let precision = tridiagonal_precision();
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let estimate = estimate_local_rbmc_variances(
            &precision,
            &factor,
            &LatentBlockMode::ContiguousPermuted { block_size: 3 },
            4,
            2,
            17,
        )
        .unwrap();
        let reference = repeated_solve_diag(&precision);
        assert!((&estimate.estimate.values - &reference).norm() < 1e-10);
        assert!(estimate
            .diagnostics
            .residual_variance_estimate
            .iter()
            .all(|value| value.abs() < 1e-12));
    }

    #[test]
    fn transformed_local_rb_rejects_split_row_assignment() {
        let precision = identity_precision(2);
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let operator =
            SparseRowOperator::new(2, vec![vec![(0, 1.0), (1, -1.0)]]).expect("valid operator");
        let err = estimate_local_rbmc_transformed_variances(
            &precision,
            &factor,
            &operator,
            &LatentBlockMode::Explicit {
                blocks: vec![vec![PermutedIndex(0)], vec![PermutedIndex(1)]],
                row_assignments: Some(vec![BlockId(0)]),
            },
            4,
            1,
            3,
        )
        .unwrap_err();
        assert!(matches!(err, GmrfError::DimensionMismatch(_)));
    }

    #[test]
    fn transformed_local_rb_accepts_overlapping_row_assigned_blocks() {
        let precision = identity_precision(3);
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let operator =
            SparseRowOperator::new(3, vec![vec![(0, 1.0), (1, -1.0)], vec![(1, 2.0), (2, 1.0)]])
                .expect("valid operator");
        let estimate = estimate_local_rbmc_transformed_variances(
            &precision,
            &factor,
            &operator,
            &LatentBlockMode::Explicit {
                blocks: vec![
                    vec![PermutedIndex(0), PermutedIndex(1)],
                    vec![PermutedIndex(1), PermutedIndex(2)],
                ],
                row_assignments: Some(vec![BlockId(0), BlockId(1)]),
            },
            8,
            2,
            13,
        )
        .unwrap();
        assert!((estimate.estimate.values[0] - 2.0).abs() < 1e-12);
        assert!((estimate.estimate.values[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn selected_inverse_transformed_matches_off_factor_row() {
        let precision = identity_precision(2);
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let operator =
            SparseRowOperator::new(2, vec![vec![(0, 1.0), (1, -1.0)]]).expect("valid operator");
        let result = selected_inverse_transformed_diag(&factor, &operator).unwrap();
        let estimate = result
            .estimate
            .expect("off-factor requested pair should be recovered by closure");
        let reference = repeated_solve_transformed_diag(&precision, &operator);
        assert!((&estimate.values - &reference).norm() < 1e-10);
        assert_eq!(result.diagnostics.requested_pairs, 3);
        assert_eq!(result.diagnostics.status, SelectedInverseStatus::Complete);
    }

    #[test]
    fn selected_inverse_entries_reports_closure_too_large() {
        let precision = identity_precision(2);
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let operator =
            SparseRowOperator::new(2, vec![vec![(0, 1.0), (1, -1.0)]]).expect("valid operator");
        let operator_tilde = operator
            .permute_columns_to_factor(&factor.permutation())
            .unwrap();
        let requested = SparseSymmetricPattern::from_transformed_operator(&operator_tilde).unwrap();
        let result = selected_inverse_entries_with_limit(&factor, &requested, 1).unwrap();
        assert!(result.inverse.is_none());
        assert_eq!(
            result.diagnostics.status,
            SelectedInverseStatus::ClosureTooLarge
        );
        assert!(result.diagnostics.closure_pairs > result.diagnostics.closure_limit);
    }

    #[test]
    fn selected_inverse_transformed_matches_diagonal_operator() {
        let precision = tridiagonal_precision();
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let operator = SparseRowOperator::identity(3);
        let result = selected_inverse_transformed_diag(&factor, &operator).unwrap();
        let estimate = result.estimate.expect("diagonal pattern should be covered");
        let reference = repeated_solve_diag(&precision);
        assert!((&estimate.values - &reference).norm() < 1e-10);
    }

    #[test]
    fn selected_inverse_transformed_sparse_row_factor_pattern_matches_repeated_solves() {
        let precision = tridiagonal_precision();
        let factor = precision.cholesky_sqrt_lower().unwrap();
        let operator = SparseRowOperator::new(
            3,
            vec![vec![(0, 1.0), (1, -1.0)], vec![(1, 1.0), (2, -1.0)]],
        )
        .expect("valid operator");
        let result = selected_inverse_transformed_diag(&factor, &operator).unwrap();
        let estimate = result
            .estimate
            .expect("factor-pattern transformed selected inverse should complete");
        let reference = repeated_solve_transformed_diag(&precision, &operator);
        assert_eq!(
            result.diagnostics.closure_pairs,
            result.diagnostics.factor_pairs
        );
        assert!((&estimate.values - &reference).norm() < 1e-10);
    }
}
