//! Observation and conditioning helpers for Gaussian observation models.
//!
//! These utilities mirror the Julia workflow of forming H^T R^-1 H and H^T R^-1 y
//! updates when conditioning a Gaussian prior on linear observations.

use crate::types::{CooMatrix, GmrfError, SparseCholeskyFactor, SparseMatrix, Vector};
use faer::get_global_parallelism;
use faer::sparse::linalg::matmul::sparse_sparse_matmul;
use faer::sparse::ops::binary_op;

/// Observation noise model for a linear Gaussian term.
#[derive(Clone, Copy, Debug)]
pub enum LinearObservationNoise<'a> {
    /// Independent observation noise with common scalar variance.
    ScalarVariance(f64),
    /// Explicit sparse observation precision on the observation rows.
    Precision(&'a SparseMatrix),
}

/// A linear Gaussian observation term `y = H x + b + eps`.
#[derive(Clone, Copy, Debug)]
pub struct LinearObservationTerm<'a> {
    pub matrix: &'a SparseMatrix,
    pub observations: &'a Vector,
    pub bias: Option<&'a Vector>,
    pub noise: LinearObservationNoise<'a>,
}

impl<'a> LinearObservationTerm<'a> {
    /// Build a term with independent scalar-variance noise.
    pub fn scalar_variance(
        matrix: &'a SparseMatrix,
        observations: &'a Vector,
        bias: Option<&'a Vector>,
        variance: f64,
    ) -> Self {
        Self {
            matrix,
            observations,
            bias,
            noise: LinearObservationNoise::ScalarVariance(variance),
        }
    }

    /// Build a term with an explicit sparse observation precision.
    pub fn precision(
        matrix: &'a SparseMatrix,
        observations: &'a Vector,
        bias: Option<&'a Vector>,
        precision: &'a SparseMatrix,
    ) -> Self {
        Self {
            matrix,
            observations,
            bias,
            noise: LinearObservationNoise::Precision(precision),
        }
    }
}

/// Sparse-work summary for one linear Gaussian observation update.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearObservationUpdateStats {
    pub operator_rows: usize,
    pub operator_cols: usize,
    pub operator_nnz: usize,
    pub precision_update_nnz: usize,
}

/// Sparse-work summary for applying a collection of linear observation terms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearObservationConditioningStats {
    pub prior_precision_nnz: usize,
    pub terms: Vec<LinearObservationUpdateStats>,
    pub posterior_precision_nnz: usize,
}

/// Fully factored posterior for a linear Gaussian conditioning update.
#[derive(Debug)]
pub struct FactoredLinearGaussianPosterior {
    pub posterior_precision: SparseMatrix,
    pub information: Vector,
    pub posterior_mean: Vector,
    pub posterior_factor: SparseCholeskyFactor,
    pub stats: LinearObservationConditioningStats,
}

/// Build an observation selector matrix that picks `indices` from a latent vector.
pub fn observation_selector(dimension: usize, indices: &[usize]) -> SparseMatrix {
    let mut coo = CooMatrix::new(indices.len(), dimension);
    for (row, idx) in indices.iter().copied().enumerate() {
        coo.push(row, idx, 1.0);
    }
    SparseMatrix::from(&coo)
}

/// Build a sparse linear observation matrix with two-point stencils per row.
///
/// Each row is defined by `(left, right, weight_right)` so that the row evaluates
/// `weight_left * x[left] + weight_right * x[right]`, where `weight_left = 1 - weight_right`.
pub fn build_linear_observation_matrix(
    dimension: usize,
    rows: &[(usize, usize, f64)],
) -> SparseMatrix {
    let mut coo = CooMatrix::new(rows.len(), dimension);
    for (row, (left, right, weight_right)) in rows.iter().enumerate() {
        let weight_right = *weight_right;
        let weight_left = 1.0 - weight_right;
        coo.push(row, *left, weight_left);
        coo.push(row, *right, weight_right);
    }
    SparseMatrix::from(&coo)
}

/// Compute H^T R^-1 y for scalar observation variance.
pub fn ht_weighted_observations(h: &SparseMatrix, y: &Vector, inv_var: f64) -> Vector {
    let mut out = Vector::zeros(h.ncols());
    for (row, col, value) in h.triplet_iter() {
        let weight = inv_var * y[row];
        out[col] += *value * weight;
    }
    out
}

/// Compute H^T R^-1 H for scalar observation variance.
pub fn ht_weighted_h(h: &SparseMatrix, inv_var: f64) -> SparseMatrix {
    let h_ref = h.as_ref();
    let h_transpose = h_ref
        .transpose()
        .to_col_major()
        .expect("failed to build H^T in column-major form");
    let htwh = sparse_sparse_matmul(
        h_transpose.as_ref(),
        h_ref,
        inv_var,
        get_global_parallelism(),
    )
    .expect("sparse-sparse matmul failed for H^T H");
    SparseMatrix::from(htwh)
}

/// Compute H^T Q y for sparse observation precision `Q`.
pub fn ht_precision_weighted_observations(
    h: &SparseMatrix,
    y: &Vector,
    precision: &SparseMatrix,
) -> Vector {
    assert_eq!(
        precision.nrows(),
        precision.ncols(),
        "observation precision must be square"
    );
    assert_eq!(
        precision.nrows(),
        h.nrows(),
        "observation precision row count must match observation operator rows"
    );
    assert_eq!(
        y.len(),
        h.nrows(),
        "observation vector length must match observation operator rows"
    );
    let weighted = precision.mul_vec(y);
    let mut out = Vector::zeros(h.ncols());
    for (row, col, value) in h.triplet_iter() {
        out[col] += *value * weighted[row];
    }
    out
}

/// Compute H^T Q H for sparse observation precision `Q`.
pub fn ht_precision_weighted_h(h: &SparseMatrix, precision: &SparseMatrix) -> SparseMatrix {
    assert_eq!(
        precision.nrows(),
        precision.ncols(),
        "observation precision must be square"
    );
    assert_eq!(
        precision.nrows(),
        h.nrows(),
        "observation precision row count must match observation operator rows"
    );
    let qh = sparse_sparse_matmul(
        precision.as_ref(),
        h.as_ref(),
        1.0,
        get_global_parallelism(),
    )
    .expect("sparse-sparse matmul failed for QH");
    let h_transpose = h
        .as_ref()
        .transpose()
        .to_col_major()
        .expect("failed to build H^T in column-major form");
    let htqh = sparse_sparse_matmul(
        h_transpose.as_ref(),
        qh.as_ref(),
        1.0,
        get_global_parallelism(),
    )
    .expect("sparse-sparse matmul failed for H^T Q H");
    SparseMatrix::from(htqh)
}

/// Add two sparse matrices, preserving duplicate entries as additive contributions.
pub fn add_sparse(a: &SparseMatrix, b: &SparseMatrix) -> SparseMatrix {
    let sum = binary_op(a.as_ref(), b.as_ref(), |lhs, rhs| {
        lhs.copied().unwrap_or(0.0) + rhs.copied().unwrap_or(0.0)
    })
    .expect("sparse add failed");
    SparseMatrix::from(sum)
}

/// Sparse observation system assembled from one or more scalar-variance terms.
#[derive(Debug, Clone)]
pub struct StackedObservationSystem {
    pub matrix: SparseMatrix,
    pub observations: Vector,
    pub bias: Vector,
    pub noise_variance: f64,
}

/// Stack scalar-variance linear observation blocks into a single whitened system.
#[derive(Debug, Clone)]
pub struct LinearObservationStackBuilder {
    dimension: usize,
    rows: Vec<(usize, usize, f64)>,
    observations: Vec<f64>,
    bias: Vec<f64>,
}

impl LinearObservationStackBuilder {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            rows: Vec::new(),
            observations: Vec::new(),
            bias: Vec::new(),
        }
    }

    pub fn push_block(
        &mut self,
        column_offset: usize,
        block: &SparseMatrix,
        observations: &[f64],
        bias: &[f64],
        variance: f64,
    ) -> Result<(), crate::GmrfError> {
        self.push_blocks(&[(column_offset, block)], observations, bias, variance)
    }

    pub fn push_blocks(
        &mut self,
        blocks: &[(usize, &SparseMatrix)],
        observations: &[f64],
        bias: &[f64],
        variance: f64,
    ) -> Result<(), crate::GmrfError> {
        if blocks.is_empty() {
            return Err(crate::GmrfError::DimensionMismatch(
                "at least one observation block is required",
            ));
        }
        let row_count = blocks[0].1.nrows();
        self.validate_rows(row_count, observations, bias, variance)?;
        let scale = variance.sqrt().recip();
        let row_offset = self.observations.len();
        for (column_offset, block) in blocks {
            if block.nrows() != row_count {
                return Err(crate::GmrfError::DimensionMismatch(
                    "all observation blocks in a term must share the same row count",
                ));
            }
            if *column_offset + block.ncols() > self.dimension {
                return Err(crate::GmrfError::DimensionMismatch(
                    "observation block exceeds the latent dimension",
                ));
            }
            for (row, col, value) in block.triplet_iter() {
                self.rows
                    .push((row_offset + row, column_offset + col, *value * scale));
            }
        }
        self.observations
            .extend(observations.iter().map(|value| *value * scale));
        self.bias.extend(bias.iter().map(|value| *value * scale));
        Ok(())
    }

    pub fn finish(self) -> StackedObservationSystem {
        let mut coo = CooMatrix::new(self.observations.len(), self.dimension);
        for (row, col, value) in self.rows {
            coo.push(row, col, value);
        }
        StackedObservationSystem {
            matrix: SparseMatrix::from(&coo),
            observations: Vector::from_vec(self.observations),
            bias: Vector::from_vec(self.bias),
            noise_variance: 1.0,
        }
    }

    fn validate_rows(
        &self,
        row_count: usize,
        observations: &[f64],
        bias: &[f64],
        variance: f64,
    ) -> Result<(), crate::GmrfError> {
        if row_count != observations.len() || row_count != bias.len() {
            return Err(crate::GmrfError::DimensionMismatch(
                "observation rows, observations, and bias lengths must match",
            ));
        }
        if !variance.is_finite() || variance <= 0.0 {
            return Err(crate::GmrfError::DimensionMismatch(
                "observation variance must be finite and positive",
            ));
        }
        Ok(())
    }
}

/// Compute the precision and information update for one linear observation term.
pub fn linear_observation_update(term: &LinearObservationTerm<'_>) -> (SparseMatrix, Vector) {
    let (precision_update, information_update, _) = linear_observation_update_with_stats(term);
    (precision_update, information_update)
}

/// Compute the precision and information update for one linear observation term,
/// returning sparse-work statistics for the same computed update.
pub fn linear_observation_update_with_stats(
    term: &LinearObservationTerm<'_>,
) -> (SparseMatrix, Vector, LinearObservationUpdateStats) {
    validate_observation_dimensions(term);
    let centered = centered_observations(term.observations, term.bias);
    let (precision_update, information_update) = match term.noise {
        LinearObservationNoise::ScalarVariance(variance) => {
            assert!(
                variance.is_finite() && variance > 0.0,
                "observation variance must be finite and positive"
            );
            let inv_var = 1.0 / variance;
            (
                ht_weighted_h(term.matrix, inv_var),
                ht_weighted_observations(term.matrix, &centered, inv_var),
            )
        }
        LinearObservationNoise::Precision(precision) => (
            ht_precision_weighted_h(term.matrix, precision),
            ht_precision_weighted_observations(term.matrix, &centered, precision),
        ),
    };
    let stats = LinearObservationUpdateStats {
        operator_rows: term.matrix.nrows(),
        operator_cols: term.matrix.ncols(),
        operator_nnz: term.matrix.nnz(),
        precision_update_nnz: precision_update.nnz(),
    };
    (precision_update, information_update, stats)
}

/// Apply a collection of linear Gaussian terms to a prior precision.
///
/// Returns `(posterior_precision, information)` where the information vector is
/// the sum of all `H^T R^-1 (y - b)` / `H^T Q (y - b)` contributions.
pub fn apply_linear_observation_terms(
    prior_precision: &SparseMatrix,
    terms: &[LinearObservationTerm<'_>],
) -> (SparseMatrix, Vector) {
    let (posterior_precision, information, _) =
        apply_linear_observation_terms_with_stats(prior_precision, terms);
    (posterior_precision, information)
}

/// Apply linear Gaussian terms and return sparse-work statistics for each
/// computed precision update.
pub fn apply_linear_observation_terms_with_stats(
    prior_precision: &SparseMatrix,
    terms: &[LinearObservationTerm<'_>],
) -> (SparseMatrix, Vector, LinearObservationConditioningStats) {
    assert_eq!(
        prior_precision.nrows(),
        prior_precision.ncols(),
        "prior precision must be square"
    );
    let mut posterior_precision = prior_precision.clone();
    let mut information = Vector::zeros(prior_precision.nrows());
    let mut term_stats = Vec::with_capacity(terms.len());
    for term in terms {
        assert_eq!(
            term.matrix.ncols(),
            prior_precision.nrows(),
            "observation operator columns must match prior dimension"
        );
        let (precision_update, information_update, stats) =
            linear_observation_update_with_stats(term);
        posterior_precision = add_sparse(&posterior_precision, &precision_update);
        information += information_update;
        term_stats.push(stats);
    }
    let stats = LinearObservationConditioningStats {
        prior_precision_nnz: prior_precision.nnz(),
        terms: term_stats,
        posterior_precision_nnz: posterior_precision.nnz(),
    };
    (posterior_precision, information, stats)
}

/// Apply linear Gaussian terms, factor the posterior precision, and solve the posterior mean.
pub fn condition_linear_gaussian_with_factor(
    prior_precision: &SparseMatrix,
    terms: &[LinearObservationTerm<'_>],
) -> Result<FactoredLinearGaussianPosterior, GmrfError> {
    let (posterior_precision, information, stats) =
        apply_linear_observation_terms_with_stats(prior_precision, terms);
    let posterior_factor = posterior_precision.cholesky_sqrt_lower()?;
    let posterior_mean = posterior_factor.solve(&information)?;
    Ok(FactoredLinearGaussianPosterior {
        posterior_precision,
        information,
        posterior_mean,
        posterior_factor,
        stats,
    })
}

/// Apply Gaussian observation conditioning in one step.
///
/// Observations follow `y = H x + b + noise`, where `b` is optional. Returns
/// `(posterior_precision, information)` with
/// `posterior_precision = prior_precision + H^T R^-1 H` and
/// `information = H^T R^-1 (y - b)`.
pub fn apply_gaussian_observations(
    prior_precision: &SparseMatrix,
    observation_matrix: &SparseMatrix,
    observations: &Vector,
    observation_bias: Option<&Vector>,
    noise_variance: f64,
) -> (SparseMatrix, Vector) {
    apply_linear_observation_terms(
        prior_precision,
        &[LinearObservationTerm::scalar_variance(
            observation_matrix,
            observations,
            observation_bias,
            noise_variance,
        )],
    )
}

/// Apply Gaussian observation conditioning with an explicit sparse observation precision.
///
/// Observations follow `y = H x + b + noise`, where the noise precision is `Q_eps`.
/// Returns `(posterior_precision, information)` with
/// `posterior_precision = prior_precision + H^T Q_eps H` and
/// `information = H^T Q_eps (y - b)`.
pub fn apply_gaussian_observations_with_precision(
    prior_precision: &SparseMatrix,
    observation_matrix: &SparseMatrix,
    observations: &Vector,
    observation_bias: Option<&Vector>,
    observation_precision: &SparseMatrix,
) -> (SparseMatrix, Vector) {
    apply_linear_observation_terms(
        prior_precision,
        &[LinearObservationTerm::precision(
            observation_matrix,
            observations,
            observation_bias,
            observation_precision,
        )],
    )
}

fn validate_observation_dimensions(term: &LinearObservationTerm<'_>) {
    assert_eq!(
        term.matrix.nrows(),
        term.observations.len(),
        "observation vector length must match observation operator rows"
    );
    if let Some(bias) = term.bias {
        assert_eq!(
            bias.len(),
            term.observations.len(),
            "observation bias length must match observations length"
        );
    }
}

fn centered_observations(observations: &Vector, bias: Option<&Vector>) -> Vector {
    match bias {
        Some(bias) => observations - bias,
        None => observations.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn assert_matrix_entries_close(lhs: &SparseMatrix, rhs: &SparseMatrix) {
        assert_eq!(lhs.nrows(), rhs.nrows());
        assert_eq!(lhs.ncols(), rhs.ncols());
        let mut entries = BTreeMap::<(usize, usize), f64>::new();
        for (row, col, value) in lhs.triplet_iter() {
            *entries.entry((row, col)).or_insert(0.0) += *value;
        }
        for (row, col, value) in rhs.triplet_iter() {
            *entries.entry((row, col)).or_insert(0.0) -= *value;
        }
        for value in entries.values() {
            assert!(value.abs() <= 1e-12, "matrix entry diff {value}");
        }
    }

    #[test]
    fn observation_update_with_stats_matches_existing_update() {
        let mut prior_coo = CooMatrix::new(3, 3);
        prior_coo.push(0, 0, 2.0);
        prior_coo.push(1, 1, 3.0);
        prior_coo.push(2, 2, 4.0);
        let prior = SparseMatrix::from(&prior_coo);

        let mut h_coo = CooMatrix::new(2, 3);
        h_coo.push(0, 0, 1.0);
        h_coo.push(0, 2, -0.5);
        h_coo.push(1, 1, 2.0);
        let h = SparseMatrix::from(&h_coo);
        let y = Vector::from_vec(vec![1.0, -2.0]);
        let bias = Vector::from_vec(vec![0.25, 0.5]);
        let terms = [LinearObservationTerm::scalar_variance(
            &h,
            &y,
            Some(&bias),
            0.2,
        )];

        let (plain_precision, plain_information) = apply_linear_observation_terms(&prior, &terms);
        let (stats_precision, stats_information, stats) =
            apply_linear_observation_terms_with_stats(&prior, &terms);

        assert_matrix_entries_close(&plain_precision, &stats_precision);
        assert!((&plain_information - &stats_information).norm() <= 1e-12);
        assert_eq!(stats.prior_precision_nnz, prior.nnz());
        assert_eq!(stats.posterior_precision_nnz, stats_precision.nnz());
        assert_eq!(stats.terms.len(), 1);
        assert_eq!(stats.terms[0].operator_rows, h.nrows());
        assert_eq!(stats.terms[0].operator_cols, h.ncols());
        assert_eq!(stats.terms[0].operator_nnz, h.nnz());
        assert!(stats.terms[0].precision_update_nnz > 0);
    }

    #[test]
    fn factored_conditioning_matches_manual_posterior_solve() {
        let mut prior_coo = CooMatrix::new(2, 2);
        prior_coo.push(0, 0, 3.0);
        prior_coo.push(1, 1, 4.0);
        let prior = SparseMatrix::from(&prior_coo);

        let mut h_coo = CooMatrix::new(1, 2);
        h_coo.push(0, 0, 1.0);
        h_coo.push(0, 1, 2.0);
        let h = SparseMatrix::from(&h_coo);
        let y = Vector::from_vec(vec![5.0]);
        let terms = [LinearObservationTerm::scalar_variance(&h, &y, None, 2.0)];

        let factored = condition_linear_gaussian_with_factor(&prior, &terms).unwrap();
        let (manual_precision, manual_information) = apply_linear_observation_terms(&prior, &terms);
        let manual_factor = manual_precision.cholesky_sqrt_lower().unwrap();
        let manual_mean = manual_factor.solve(&manual_information).unwrap();

        assert_matrix_entries_close(&factored.posterior_precision, &manual_precision);
        assert!((&factored.information - &manual_information).norm() <= 1e-12);
        assert!((&factored.posterior_mean - &manual_mean).norm() <= 1e-12);
        assert_eq!(
            factored.stats.posterior_precision_nnz,
            factored.posterior_precision.nnz()
        );
    }
}
