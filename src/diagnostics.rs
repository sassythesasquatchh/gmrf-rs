//! Generic diagnostics for Gaussian predictive distributions.

use crate::GmrfError;

/// Convert a finite symmetric covariance matrix to a correlation matrix.
///
/// Rows and columns with zero variance are returned as all zeros. Tiny negative
/// diagonal entries caused by floating-point roundoff are treated as zero;
/// materially negative diagonal entries are rejected.
pub fn covariance_to_correlation(covariance: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GmrfError> {
    let dimension = covariance.len();
    if dimension == 0 || covariance.iter().any(|row| row.len() != dimension) {
        return Err(GmrfError::DimensionMismatch(
            "covariance matrix must be non-empty and square",
        ));
    }
    if covariance.iter().flatten().any(|value| !value.is_finite()) {
        return Err(GmrfError::NumericalInstability(
            "covariance matrix entries must be finite",
        ));
    }

    let scale = covariance
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    let tolerance = 1.0e-10 * scale;
    for (row_index, row) in covariance.iter().enumerate() {
        for (column_index, column) in covariance.iter().enumerate().take(row_index) {
            if (row[column_index] - column[row_index]).abs() > tolerance {
                return Err(GmrfError::NumericalInstability(
                    "covariance matrix must be symmetric within numerical tolerance",
                ));
            }
        }
    }

    let diagonal_tolerance = 1.0e-12 * scale;
    let mut standard_deviations = Vec::with_capacity(dimension);
    for (index, row) in covariance.iter().enumerate() {
        let variance = row[index];
        if variance < -diagonal_tolerance {
            return Err(GmrfError::NumericalInstability(
                "covariance matrix has a materially negative diagonal entry",
            ));
        }
        standard_deviations.push(variance.max(0.0).sqrt());
    }

    let mut correlation = vec![vec![0.0; dimension]; dimension];
    for row in 0..dimension {
        if standard_deviations[row] == 0.0 {
            continue;
        }
        for column in 0..dimension {
            if standard_deviations[column] != 0.0 {
                correlation[row][column] = covariance[row][column]
                    / (standard_deviations[row] * standard_deviations[column]);
            }
        }
        correlation[row][row] = 1.0;
    }
    Ok(correlation)
}

/// Pointwise and aggregate diagnostics for independent Gaussian predictions.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianPredictiveDiagnostics {
    pub residuals: Vec<f64>,
    pub predictive_variances: Vec<f64>,
    pub predictive_standard_deviations: Vec<f64>,
    pub standardized_residuals: Vec<f64>,
    pub rmse: f64,
    pub mean_negative_log_predictive_density: f64,
    pub interval_standard_deviations: f64,
    pub empirical_coverage: f64,
}

/// Evaluate independent Gaussian predictions with latent and observation variance.
///
/// The predictive variance in row `i` is
/// `latent_variances[i] + observation_variances[i]`.
pub fn gaussian_predictive_diagnostics(
    observations: &[f64],
    predictive_means: &[f64],
    latent_variances: &[f64],
    observation_variances: &[f64],
    interval_standard_deviations: f64,
) -> Result<GaussianPredictiveDiagnostics, GmrfError> {
    let dimension = observations.len();
    if dimension == 0
        || predictive_means.len() != dimension
        || latent_variances.len() != dimension
        || observation_variances.len() != dimension
    {
        return Err(GmrfError::DimensionMismatch(
            "predictive diagnostic vectors must be non-empty and have matching lengths",
        ));
    }
    if !interval_standard_deviations.is_finite() || interval_standard_deviations <= 0.0 {
        return Err(GmrfError::NumericalInstability(
            "predictive interval width must be finite and positive",
        ));
    }
    if observations
        .iter()
        .chain(predictive_means)
        .chain(latent_variances)
        .chain(observation_variances)
        .any(|value| !value.is_finite())
        || latent_variances.iter().any(|variance| *variance < 0.0)
        || observation_variances
            .iter()
            .any(|variance| *variance <= 0.0)
    {
        return Err(GmrfError::NumericalInstability(
            "predictive inputs must be finite with non-negative latent and positive observation variances",
        ));
    }

    let mut residuals = Vec::with_capacity(dimension);
    let mut predictive_variances = Vec::with_capacity(dimension);
    let mut predictive_standard_deviations = Vec::with_capacity(dimension);
    let mut standardized_residuals = Vec::with_capacity(dimension);
    let mut squared_error = 0.0;
    let mut nlpd = 0.0;
    let mut covered = 0_usize;

    for index in 0..dimension {
        let residual = observations[index] - predictive_means[index];
        let predictive_variance = latent_variances[index] + observation_variances[index];
        if !predictive_variance.is_finite() || predictive_variance <= 0.0 {
            return Err(GmrfError::NumericalInstability(
                "predictive variance must be finite and positive",
            ));
        }
        let predictive_standard_deviation = predictive_variance.sqrt();
        let standardized_residual = residual / predictive_standard_deviation;
        squared_error += residual * residual;
        nlpd += 0.5
            * ((2.0 * std::f64::consts::PI * predictive_variance).ln()
                + standardized_residual * standardized_residual);
        if standardized_residual.abs() <= interval_standard_deviations {
            covered += 1;
        }
        residuals.push(residual);
        predictive_variances.push(predictive_variance);
        predictive_standard_deviations.push(predictive_standard_deviation);
        standardized_residuals.push(standardized_residual);
    }

    Ok(GaussianPredictiveDiagnostics {
        residuals,
        predictive_variances,
        predictive_standard_deviations,
        standardized_residuals,
        rmse: (squared_error / dimension as f64).sqrt(),
        mean_negative_log_predictive_density: nlpd / dimension as f64,
        interval_standard_deviations,
        empirical_coverage: covered as f64 / dimension as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictive_diagnostics_match_known_independent_values() {
        let diagnostics = gaussian_predictive_diagnostics(
            &[1.0, -1.0],
            &[0.0, 0.0],
            &[0.75, 0.75],
            &[0.25, 0.25],
            1.96,
        )
        .unwrap();
        assert_eq!(diagnostics.residuals, [1.0, -1.0]);
        assert_eq!(diagnostics.predictive_variances, [1.0, 1.0]);
        assert_eq!(diagnostics.standardized_residuals, [1.0, -1.0]);
        assert!((diagnostics.rmse - 1.0).abs() < 1.0e-12);
        assert_eq!(diagnostics.empirical_coverage, 1.0);
    }

    #[test]
    fn predictive_diagnostics_reject_invalid_variances() {
        assert!(gaussian_predictive_diagnostics(&[0.0], &[0.0], &[-1.0], &[1.0], 1.96).is_err());
        assert!(gaussian_predictive_diagnostics(&[0.0], &[0.0], &[0.0], &[0.0], 1.96).is_err());
    }

    #[test]
    fn covariance_to_correlation_handles_positive_and_zero_variance_rows() {
        let correlation = covariance_to_correlation(&[
            vec![4.0, 2.0, 0.0],
            vec![2.0, 9.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ])
        .unwrap();
        assert!((correlation[0][1] - 1.0 / 3.0).abs() < 1.0e-12);
        assert_eq!(correlation[0][0], 1.0);
        assert_eq!(correlation[2], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn covariance_to_correlation_rejects_invalid_covariances() {
        assert!(covariance_to_correlation(&[vec![1.0, 0.5], vec![0.0, 1.0]]).is_err());
        assert!(covariance_to_correlation(&[vec![f64::NAN]]).is_err());
        assert!(covariance_to_correlation(&[vec![-1.0]]).is_err());
    }

    #[test]
    fn covariance_to_correlation_accepts_roundoff_negative_diagonal() {
        assert_eq!(
            covariance_to_correlation(&[vec![-1.0e-14]]).unwrap(),
            [vec![0.0]]
        );
    }
}
