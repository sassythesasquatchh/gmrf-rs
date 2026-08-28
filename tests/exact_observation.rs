use gmrf_core::{
    condition_linear_gaussian_with_factor, types::CooMatrix, DenseMatrix, GmrfError,
    LinearObservationTerm, PreparedExactObservationPosterior, SparseMatrix, Vector,
};

const COMPARISON_TOLERANCE: f64 = 2.0e-12;

#[test]
fn covariance_block_conditioner_matches_direct_dense_formula_and_reuses_factor() {
    let kyy = DenseMatrix::from_fn(2, 2, |row, column| [[2.0, 0.3], [0.3, 1.5]][row][column]);
    let kgy = DenseMatrix::from_fn(2, 2, |row, column| [[0.5, -0.2], [1.0, 0.4]][row][column]);
    let kgg = DenseMatrix::from_fn(2, 2, |row, column| [[1.2, 0.1], [0.1, 2.0]][row][column]);
    let noise_variance = 0.25;
    let prepared = PreparedExactObservationPosterior::from_prior_covariance_blocks(
        &kyy,
        &kgy,
        &kgg,
        noise_variance,
    )
    .unwrap();

    let observations = Vector::from_vec(vec![1.2, -0.7]);
    let determinant = (2.0 + noise_variance) * (1.5 + noise_variance) - 0.3 * 0.3;
    let solved = [
        ((1.5 + noise_variance) * observations[0] - 0.3 * observations[1]) / determinant,
        (-0.3 * observations[0] + (2.0 + noise_variance) * observations[1]) / determinant,
    ];
    let expected_mean = [
        kgy[(0, 0)] * solved[0] + kgy[(0, 1)] * solved[1],
        kgy[(1, 0)] * solved[0] + kgy[(1, 1)] * solved[1],
    ];
    assert_vector_close(
        &prepared.output_mean(&observations).unwrap(),
        &expected_mean,
    );

    let inverse = [
        [(1.5 + noise_variance) / determinant, -0.3 / determinant],
        [-0.3 / determinant, (2.0 + noise_variance) / determinant],
    ];
    let expected_covariance = DenseMatrix::from_fn(2, 2, |row, column| {
        let removed = (0..2)
            .map(|left| {
                (0..2)
                    .map(|right| kgy[(row, left)] * inverse[left][right] * kgy[(column, right)])
                    .sum::<f64>()
            })
            .sum::<f64>();
        kgg[(row, column)] - removed
    });
    assert_matrix_close(prepared.posterior_output_covariance(), &expected_covariance);

    let prior_observation_mean = Vector::from_vec(vec![0.2, -0.1]);
    let prior_output_mean = Vector::from_vec(vec![-0.4, 0.6]);
    let centered = [
        observations[0] - prior_observation_mean[0],
        observations[1] - prior_observation_mean[1],
    ];
    let centered_solved = [
        ((1.5 + noise_variance) * centered[0] - 0.3 * centered[1]) / determinant,
        (-0.3 * centered[0] + (2.0 + noise_variance) * centered[1]) / determinant,
    ];
    let expected_nonzero_prior_mean = [
        prior_output_mean[0] + kgy[(0, 0)] * centered_solved[0] + kgy[(0, 1)] * centered_solved[1],
        prior_output_mean[1] + kgy[(1, 0)] * centered_solved[0] + kgy[(1, 1)] * centered_solved[1],
    ];
    assert_vector_close(
        &prepared
            .output_mean_with_prior(&observations, &prior_observation_mean, &prior_output_mean)
            .unwrap(),
        &expected_nonzero_prior_mean,
    );

    let covariance_input = Vector::from_vec(vec![0.7, -0.2]);
    let expected_covariance_action = Vector::from_fn(2, |row| {
        (0..2)
            .map(|column| expected_covariance[(row, column)] * covariance_input[column])
            .sum()
    });
    assert_vector_close(
        &prepared
            .posterior_output_covariance_action(&covariance_input)
            .unwrap(),
        expected_covariance_action.as_slice(),
    );

    let replacement = Vector::from_vec(vec![-0.4, 2.1]);
    let replacement_mean = prepared.output_mean(&replacement).unwrap();
    assert_ne!(
        replacement_mean,
        prepared.output_mean(&observations).unwrap()
    );
    assert_matrix_close(prepared.posterior_output_covariance(), &expected_covariance);

    let diagnostics = prepared.diagnostics();
    assert_eq!(diagnostics.observation_count, 2);
    assert_eq!(diagnostics.output_count, 2);
    assert_eq!(diagnostics.dense_factor_dimension, 2);
    assert_eq!(diagnostics.dense_factor_allocated_entries, 4);
    assert_eq!(diagnostics.dense_factor_triangular_entries, 3);
    assert_eq!(diagnostics.output_observation_covariance_entries, 4);
    assert_eq!(diagnostics.posterior_output_covariance_entries, 4);
    assert_eq!(diagnostics.persistent_f64_entries, 12);
    assert!(diagnostics
        .posterior_output_covariance_max_abs_skew
        .is_finite());
}

#[test]
fn exact_observation_conditioner_matches_sparse_precision_conditioning() {
    let precision_dense = [[2.0, -0.5, 0.0], [-0.5, 1.5, -0.25], [0.0, -0.25, 1.25]];
    let prior_precision = sparse_from_dense(&precision_dense);
    let prior_factor = prior_precision.cholesky_sqrt_lower().unwrap();
    let prior_covariance = DenseMatrix::from_fn(3, 3, |row, column| {
        let rhs = Vector::from_fn(3, |index| f64::from(index == column));
        prior_factor.solve(&rhs).unwrap()[row]
    });

    let h_dense = [[1.0, 0.0, 0.5], [0.0, 1.0, -0.25]];
    let g_dense = [[0.2, -0.4, 1.0], [1.0, 0.5, 0.0]];
    let h = sparse_from_dense(&h_dense);
    // Independent sparse covariance actions are not bitwise symmetric. The
    // application-level assembler must audit them before this explicit
    // projection; the generic conditioner intentionally applies no tolerance.
    let kyy = symmetric_projection(&covariance_pushforward(
        &h_dense,
        &prior_covariance,
        &h_dense,
    ));
    let kgy = covariance_pushforward(&g_dense, &prior_covariance, &h_dense);
    let kgg = symmetric_projection(&covariance_pushforward(
        &g_dense,
        &prior_covariance,
        &g_dense,
    ));
    let noise_variance = 0.3;
    let prepared = PreparedExactObservationPosterior::from_prior_covariance_blocks(
        &kyy,
        &kgy,
        &kgg,
        noise_variance,
    )
    .unwrap();

    for observations in [
        Vector::from_vec(vec![0.8, -1.1]),
        Vector::from_vec(vec![-0.3, 0.4]),
    ] {
        let sparse_posterior = condition_linear_gaussian_with_factor(
            &prior_precision,
            &[LinearObservationTerm::scalar_variance(
                &h,
                &observations,
                None,
                noise_variance,
            )],
        )
        .unwrap();
        let expected_output_mean = Vector::from_fn(2, |row| {
            (0..3)
                .map(|column| g_dense[row][column] * sparse_posterior.posterior_mean[column])
                .sum()
        });
        assert_vector_close(
            &prepared.output_mean(&observations).unwrap(),
            expected_output_mean.as_slice(),
        );

        let expected_output_covariance = DenseMatrix::from_fn(2, 2, |row, column| {
            let rhs = Vector::from_fn(3, |state| g_dense[column][state]);
            let covariance_action = sparse_posterior.posterior_factor.solve(&rhs).unwrap();
            (0..3)
                .map(|state| g_dense[row][state] * covariance_action[state])
                .sum()
        });
        assert_matrix_close(
            prepared.posterior_output_covariance(),
            &expected_output_covariance,
        );
    }
}

#[test]
fn scalar_noise_and_already_noisy_constructors_are_equivalent() {
    let kyy = DenseMatrix::from_fn(2, 2, |row, column| [[1.4, 0.2], [0.2, 0.9]][row][column]);
    let noise_variance = 0.3;
    let noisy_kyy = DenseMatrix::from_fn(2, 2, |row, column| {
        kyy[(row, column)] + f64::from(row == column) * noise_variance
    });
    let kgy = DenseMatrix::from_fn(1, 2, |_, column| [0.6, -0.1][column]);
    let kgg = DenseMatrix::from_fn(1, 1, |_, _| 1.1);

    let explicit = PreparedExactObservationPosterior::from_prior_covariance_blocks(
        &kyy,
        &kgy,
        &kgg,
        noise_variance,
    )
    .unwrap();
    let already_noisy =
        PreparedExactObservationPosterior::from_noisy_covariance_blocks(&noisy_kyy, &kgy, &kgg)
            .unwrap();
    let observations = Vector::from_vec(vec![0.3, 0.7]);

    assert_vector_close(
        &explicit.output_mean(&observations).unwrap(),
        already_noisy.output_mean(&observations).unwrap().as_slice(),
    );
    assert_matrix_close(
        explicit.posterior_output_covariance(),
        already_noisy.posterior_output_covariance(),
    );
}

#[test]
fn exact_observation_conditioner_rejects_ambiguous_or_invalid_inputs() {
    let nonsymmetric =
        DenseMatrix::from_fn(2, 2, |row, column| [[1.0, 0.2], [0.1, 1.0]][row][column]);
    let kgy = DenseMatrix::from_fn(1, 2, |_, column| [0.5, 0.25][column]);
    let kgg = DenseMatrix::identity(1, 1);
    assert!(matches!(
        PreparedExactObservationPosterior::from_noisy_covariance_blocks(&nonsymmetric, &kgy, &kgg),
        Err(GmrfError::NonSymmetricGaussianCovariance("Kyy"))
    ));

    let symmetric = DenseMatrix::identity(2, 2);
    assert!(matches!(
        PreparedExactObservationPosterior::from_prior_covariance_blocks(
            &symmetric, &kgy, &kgg, -1.0
        ),
        Err(GmrfError::InvalidObservationNoiseVariance)
    ));

    let wrong_kgy = DenseMatrix::zeros(1, 3);
    assert!(matches!(
        PreparedExactObservationPosterior::from_noisy_covariance_blocks(
            &symmetric, &wrong_kgy, &kgg
        ),
        Err(GmrfError::DimensionMismatch(_))
    ));

    let nonfinite_kgy =
        DenseMatrix::from_fn(1, 2, |_, column| if column == 0 { f64::NAN } else { 0.0 });
    assert!(matches!(
        PreparedExactObservationPosterior::from_noisy_covariance_blocks(
            &symmetric,
            &nonfinite_kgy,
            &kgg
        ),
        Err(GmrfError::NonFiniteGaussianInput("Kgy"))
    ));
}

fn sparse_from_dense<const ROWS: usize, const COLUMNS: usize>(
    dense: &[[f64; COLUMNS]; ROWS],
) -> SparseMatrix {
    let mut coo = CooMatrix::new(ROWS, COLUMNS);
    for (row, values) in dense.iter().enumerate() {
        for (column, value) in values.iter().copied().enumerate() {
            if value != 0.0 {
                coo.push(row, column, value);
            }
        }
    }
    SparseMatrix::from(&coo)
}

fn covariance_pushforward<const LEFT: usize, const STATE: usize, const RIGHT: usize>(
    left: &[[f64; STATE]; LEFT],
    covariance: &DenseMatrix,
    right: &[[f64; STATE]; RIGHT],
) -> DenseMatrix {
    DenseMatrix::from_fn(LEFT, RIGHT, |row, column| {
        (0..STATE)
            .map(|left_state| {
                (0..STATE)
                    .map(|right_state| {
                        left[row][left_state]
                            * covariance[(left_state, right_state)]
                            * right[column][right_state]
                    })
                    .sum::<f64>()
            })
            .sum()
    })
}

fn symmetric_projection(matrix: &DenseMatrix) -> DenseMatrix {
    DenseMatrix::from_fn(matrix.nrows(), matrix.ncols(), |row, column| {
        0.5 * (matrix[(row, column)] + matrix[(column, row)])
    })
}

fn assert_vector_close(actual: &Vector, expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= COMPARISON_TOLERANCE,
            "vector mismatch: {actual} versus {expected}"
        );
    }
}

fn assert_matrix_close(actual: &DenseMatrix, expected: &DenseMatrix) {
    assert_eq!(actual.nrows(), expected.nrows());
    assert_eq!(actual.ncols(), expected.ncols());
    for row in 0..actual.nrows() {
        for column in 0..actual.ncols() {
            assert!(
                (actual[(row, column)] - expected[(row, column)]).abs() <= COMPARISON_TOLERANCE,
                "matrix mismatch at ({row}, {column}): {} versus {}",
                actual[(row, column)],
                expected[(row, column)]
            );
        }
    }
}
