//! Exact Gaussian conditioning from dense covariance blocks.
//!
//! This module owns the generic observation-space algebra for an exact prior
//! covariance `C`, observation map `H`, and output map `G`.  An application
//! supplies the blocks
//!
//! ```text
//! Kyy = H C H^T,
//! Kgy = G C H^T,
//! Kgg = G C G^T.
//! ```
//!
//! The prepared conditioner factors `Kyy + R` once. Replacement data reuse
//! that factor, while the posterior output covariance is independent of the
//! observed values. No FEEC, geometry, or case-study assumptions enter here.

use crate::types::DenseMatrix;
use crate::{GmrfError, Vector};
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::{Llt, Solve};
use faer::{Accum, Par, Side};

/// Persistent dimensions and dense floating-point storage owned by an exact
/// observation-space conditioner.
///
/// The Cholesky implementation owns a dense square backing allocation even
/// though only its lower triangle is mathematically active. The counts exclude
/// constructor/query temporaries and covariance blocks retained by the caller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactObservationPosteriorDiagnostics {
    pub observation_count: usize,
    pub output_count: usize,
    pub dense_factor_dimension: usize,
    pub dense_factor_allocated_entries: usize,
    pub dense_factor_triangular_entries: usize,
    pub output_observation_covariance_entries: usize,
    pub posterior_output_covariance_entries: usize,
    pub persistent_f64_entries: usize,
    /// Maximum absolute numerical skew in the computed posterior output block.
    /// No tolerance is applied and the block is not silently symmetrized.
    pub posterior_output_covariance_max_abs_skew: f64,
}

/// Exact Gaussian posterior prepared from observation/output covariance blocks.
///
/// The object stores the dense factor of `Kyy + R`, `Kgy`, and the already
/// computed posterior output covariance
/// `Kgg - Kgy (Kyy + R)^-1 Kyg`. It does not retain a latent covariance.
#[derive(Debug, Clone)]
pub struct PreparedExactObservationPosterior {
    factor: Llt<f64>,
    output_observation_covariance: DenseMatrix,
    posterior_output_covariance: DenseMatrix,
    posterior_output_covariance_max_abs_skew: f64,
}

impl PreparedExactObservationPosterior {
    /// Prepare an exact observation-space posterior from prior covariance blocks.
    ///
    /// `prior_observation_covariance` is `Kyy` before observation noise,
    /// `output_observation_covariance` is `Kgy` with shape `q x m`, and
    /// `prior_output_covariance` is `Kgg` with shape `q x q`.
    pub fn from_prior_covariance_blocks(
        prior_observation_covariance: &DenseMatrix,
        output_observation_covariance: &DenseMatrix,
        prior_output_covariance: &DenseMatrix,
        noise_variance: f64,
    ) -> Result<Self, GmrfError> {
        validate_covariance_blocks(
            prior_observation_covariance,
            output_observation_covariance,
            prior_output_covariance,
        )?;

        if !noise_variance.is_finite() || noise_variance < 0.0 {
            return Err(GmrfError::InvalidObservationNoiseVariance);
        }
        let mut noisy_observation_covariance = prior_observation_covariance.clone();
        for index in 0..noisy_observation_covariance.nrows() {
            noisy_observation_covariance[(index, index)] += noise_variance;
        }

        Self::from_noisy_covariance_blocks(
            &noisy_observation_covariance,
            output_observation_covariance,
            prior_output_covariance,
        )
    }

    /// Prepare from an observation covariance that already includes noise.
    ///
    /// This constructor exists to make the `Kyy + R` convention explicit for
    /// callers that construct the noisy block externally. It never adds noise.
    pub fn from_noisy_covariance_blocks(
        noisy_observation_covariance: &DenseMatrix,
        output_observation_covariance: &DenseMatrix,
        prior_output_covariance: &DenseMatrix,
    ) -> Result<Self, GmrfError> {
        validate_covariance_blocks(
            noisy_observation_covariance,
            output_observation_covariance,
            prior_output_covariance,
        )?;

        let factor = noisy_observation_covariance
            .clone()
            .llt(Side::Lower)
            .map_err(|_| GmrfError::NonPositiveDefinite)?;

        let observation_count = noisy_observation_covariance.nrows();
        let output_count = output_observation_covariance.nrows();
        let mut solved_cross_covariance =
            DenseMatrix::from_fn(observation_count, output_count, |row, column| {
                output_observation_covariance[(column, row)]
            });
        factor.solve_in_place(solved_cross_covariance.as_mut());

        let mut removed_covariance = DenseMatrix::zeros(output_count, output_count);
        matmul(
            &mut removed_covariance,
            Accum::Replace,
            output_observation_covariance.as_ref(),
            solved_cross_covariance.as_ref(),
            1.0,
            Par::Seq,
        );
        let posterior_output_covariance =
            DenseMatrix::from_fn(output_count, output_count, |row, column| {
                prior_output_covariance[(row, column)] - removed_covariance[(row, column)]
            });
        let posterior_output_covariance_max_abs_skew =
            maximum_absolute_skew(&posterior_output_covariance);

        Ok(Self {
            factor,
            output_observation_covariance: output_observation_covariance.clone(),
            posterior_output_covariance,
            posterior_output_covariance_max_abs_skew,
        })
    }

    pub fn observation_count(&self) -> usize {
        self.output_observation_covariance.ncols()
    }

    pub fn output_count(&self) -> usize {
        self.output_observation_covariance.nrows()
    }

    /// Posterior output mean for a zero-mean prior and replacement observations.
    pub fn output_mean(&self, observations: &Vector) -> Result<Vector, GmrfError> {
        if observations.len() != self.observation_count() {
            return Err(GmrfError::DimensionMismatch(
                "observation values must match Kyy",
            ));
        }
        validate_finite_vector(observations, "observation values")?;
        self.output_mean_from_centered(observations, None)
    }

    /// Posterior output mean with explicit prior observation/output means.
    ///
    /// This evaluates
    /// `mu_g + Kgy (Kyy + R)^-1 (y - mu_y)` without refactorization.
    pub fn output_mean_with_prior(
        &self,
        observations: &Vector,
        prior_observation_mean: &Vector,
        prior_output_mean: &Vector,
    ) -> Result<Vector, GmrfError> {
        if observations.len() != self.observation_count()
            || prior_observation_mean.len() != self.observation_count()
        {
            return Err(GmrfError::DimensionMismatch(
                "observation values and prior observation mean must match Kyy",
            ));
        }
        if prior_output_mean.len() != self.output_count() {
            return Err(GmrfError::DimensionMismatch(
                "prior output mean must match Kgg",
            ));
        }
        validate_finite_vector(observations, "observation values")?;
        validate_finite_vector(prior_observation_mean, "prior observation mean")?;
        validate_finite_vector(prior_output_mean, "prior output mean")?;

        let centered = Vector::from_fn(self.observation_count(), |row| {
            observations[row] - prior_observation_mean[row]
        });
        self.output_mean_from_centered(&centered, Some(prior_output_mean))
    }

    fn output_mean_from_centered(
        &self,
        centered: &Vector,
        prior_output_mean: Option<&Vector>,
    ) -> Result<Vector, GmrfError> {
        let mut centered =
            DenseMatrix::from_fn(self.observation_count(), 1, |row, _| centered[row]);
        self.factor.solve_in_place(centered.as_mut());

        let mut correction = DenseMatrix::zeros(self.output_count(), 1);
        matmul(
            &mut correction,
            Accum::Replace,
            self.output_observation_covariance.as_ref(),
            centered.as_ref(),
            1.0,
            Par::Seq,
        );
        Ok(Vector::from_fn(self.output_count(), |row| {
            prior_output_mean.map_or(0.0, |mean| mean[row]) + correction[(row, 0)]
        }))
    }

    /// Borrow the full posterior covariance of the requested outputs.
    pub fn posterior_output_covariance(&self) -> &DenseMatrix {
        &self.posterior_output_covariance
    }

    /// Apply the posterior output covariance without exposing its storage layout.
    pub fn posterior_output_covariance_action(&self, input: &Vector) -> Result<Vector, GmrfError> {
        if input.len() != self.output_count() {
            return Err(GmrfError::DimensionMismatch(
                "output covariance input must match Kgg",
            ));
        }
        validate_finite_vector(input, "output covariance input")?;
        Ok(Vector::from_fn(self.output_count(), |row| {
            (0..self.output_count())
                .map(|column| self.posterior_output_covariance[(row, column)] * input[column])
                .sum()
        }))
    }

    pub fn diagnostics(&self) -> ExactObservationPosteriorDiagnostics {
        let observation_count = self.observation_count();
        let output_count = self.output_count();
        let dense_factor_allocated_entries = observation_count * observation_count;
        let output_observation_covariance_entries = output_count * observation_count;
        let posterior_output_covariance_entries = output_count * output_count;
        ExactObservationPosteriorDiagnostics {
            observation_count,
            output_count,
            dense_factor_dimension: observation_count,
            dense_factor_allocated_entries,
            dense_factor_triangular_entries: observation_count * (observation_count + 1) / 2,
            output_observation_covariance_entries,
            posterior_output_covariance_entries,
            persistent_f64_entries: dense_factor_allocated_entries
                + output_observation_covariance_entries
                + posterior_output_covariance_entries,
            posterior_output_covariance_max_abs_skew: self.posterior_output_covariance_max_abs_skew,
        }
    }
}

fn validate_covariance_blocks(
    observation_covariance: &DenseMatrix,
    output_observation_covariance: &DenseMatrix,
    output_covariance: &DenseMatrix,
) -> Result<(), GmrfError> {
    validate_symmetric_finite_covariance(observation_covariance, "Kyy")?;
    validate_symmetric_finite_covariance(output_covariance, "Kgg")?;
    if observation_covariance.nrows() == 0 {
        return Err(GmrfError::DimensionMismatch(
            "exact observation conditioner requires at least one observation",
        ));
    }
    if output_covariance.nrows() == 0 {
        return Err(GmrfError::DimensionMismatch(
            "exact observation conditioner requires at least one output",
        ));
    }
    if output_observation_covariance.ncols() != observation_covariance.nrows()
        || output_observation_covariance.nrows() != output_covariance.nrows()
    {
        return Err(GmrfError::DimensionMismatch(
            "Kgy dimensions must be output count by observation count",
        ));
    }
    validate_finite_matrix(output_observation_covariance, "Kgy")
}

fn validate_symmetric_finite_covariance(
    covariance: &DenseMatrix,
    label: &'static str,
) -> Result<(), GmrfError> {
    if covariance.nrows() != covariance.ncols() {
        return Err(GmrfError::DimensionMismatch(
            "dense Gaussian covariance block must be square",
        ));
    }
    validate_finite_matrix(covariance, label)?;
    for row in 0..covariance.nrows() {
        for column in 0..row {
            if covariance[(row, column)] != covariance[(column, row)] {
                return Err(GmrfError::NonSymmetricGaussianCovariance(label));
            }
        }
    }
    Ok(())
}

fn validate_finite_matrix(matrix: &DenseMatrix, label: &'static str) -> Result<(), GmrfError> {
    if (0..matrix.nrows())
        .any(|row| (0..matrix.ncols()).any(|column| !matrix[(row, column)].is_finite()))
    {
        return Err(GmrfError::NonFiniteGaussianInput(label));
    }
    Ok(())
}

fn validate_finite_vector(vector: &Vector, label: &'static str) -> Result<(), GmrfError> {
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(GmrfError::NonFiniteGaussianInput(label));
    }
    Ok(())
}

fn maximum_absolute_skew(matrix: &DenseMatrix) -> f64 {
    let mut maximum = 0.0_f64;
    for row in 0..matrix.nrows() {
        for column in 0..row {
            maximum = maximum.max((matrix[(row, column)] - matrix[(column, row)]).abs());
        }
    }
    maximum
}
