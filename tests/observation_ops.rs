use gmrf_core::observation::{
    apply_gaussian_observations, apply_gaussian_observations_with_precision,
    apply_linear_observation_terms, ht_precision_weighted_h, ht_weighted_h,
    LinearObservationStackBuilder, LinearObservationTerm,
};
use gmrf_core::types::{CooMatrix, SparseMatrix, Vector};

fn dense_from_sparse(mat: &SparseMatrix) -> Vec<Vec<f64>> {
    let mut dense = vec![vec![0.0; mat.ncols()]; mat.nrows()];
    for (row, col, val) in mat.triplet_iter() {
        dense[row][col] += *val;
    }
    dense
}

fn assert_dense_close(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
    assert_eq!(actual.len(), expected.len());
    for (row_idx, (row_a, row_e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(row_a.len(), row_e.len());
        for (col_idx, (a, e)) in row_a.iter().zip(row_e.iter()).enumerate() {
            let diff = (a - e).abs();
            assert!(
                diff < 1e-12,
                "mismatch at ({}, {}): {} vs {}",
                row_idx,
                col_idx,
                a,
                e
            );
        }
    }
}

#[test]
fn observations_use_sparse_ops_and_match_manual() {
    let mut coo = CooMatrix::new(3, 4);
    // Row 0: [1.0, 0.0, 2.0, 0.0]
    coo.push(0, 0, 1.0);
    coo.push(0, 2, 2.0);
    // Row 1: [0.0, 3.0, 0.0, 0.0]
    coo.push(1, 1, 3.0);
    // Row 2: [-1.0, 0.0, 0.0, 0.5]
    coo.push(2, 0, -1.0);
    coo.push(2, 3, 0.5);
    let h = SparseMatrix::from(&coo);

    let noise_variance = 2.0;
    let inv_var = 1.0 / noise_variance;
    let observations = Vector::from_vec(vec![1.0, -2.0, 0.5]);

    let mut prior_coo = CooMatrix::new(4, 4);
    prior_coo.push(0, 0, 1.0);
    prior_coo.push(1, 1, 2.0);
    prior_coo.push(2, 2, 3.0);
    prior_coo.push(3, 3, 4.0);
    let prior = SparseMatrix::from(&prior_coo);

    let (posterior, info) =
        apply_gaussian_observations(&prior, &h, &observations, None, noise_variance);

    let mut expected_info = [0.0; 4];
    for (row, col, value) in h.triplet_iter() {
        expected_info[col] += inv_var * value * observations[row];
    }

    for (idx, (actual, expected)) in info.iter().zip(expected_info.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-12,
            "info mismatch at {}: {} vs {}",
            idx,
            actual,
            expected
        );
    }

    let htwh = ht_weighted_h(&h, inv_var);
    let mut expected_htwh = vec![vec![0.0; 4]; 4];
    let rows = vec![
        vec![(0, 1.0), (2, 2.0)],
        vec![(1, 3.0)],
        vec![(0, -1.0), (3, 0.5)],
    ];
    for entries in rows {
        for (col_i, val_i) in entries.iter() {
            for (col_j, val_j) in entries.iter() {
                expected_htwh[*col_i][*col_j] += inv_var * val_i * val_j;
            }
        }
    }

    let mut expected_posterior = dense_from_sparse(&prior);
    for i in 0..4 {
        for j in 0..4 {
            expected_posterior[i][j] += expected_htwh[i][j];
        }
    }

    let actual_posterior = dense_from_sparse(&posterior);
    assert_dense_close(&actual_posterior, &expected_posterior);

    let actual_htwh = dense_from_sparse(&htwh);
    assert_dense_close(&actual_htwh, &expected_htwh);
}

#[test]
fn observations_support_affine_bias() {
    let mut coo = CooMatrix::new(2, 2);
    coo.push(0, 0, 1.0);
    coo.push(1, 1, 2.0);
    let h = SparseMatrix::from(&coo);

    let mut prior_coo = CooMatrix::new(2, 2);
    prior_coo.push(0, 0, 0.5);
    prior_coo.push(1, 1, 0.25);
    let prior = SparseMatrix::from(&prior_coo);

    let observations = Vector::from_vec(vec![1.5, -2.0]);
    let bias = Vector::from_vec(vec![0.5, -1.0]);
    let noise_variance = 2.0;
    let inv_var = 1.0 / noise_variance;

    let (posterior, info) =
        apply_gaussian_observations(&prior, &h, &observations, Some(&bias), noise_variance);

    let expected_info = [inv_var * 1.0, inv_var * -2.0];
    for (idx, (actual, expected)) in info.iter().zip(expected_info.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-12,
            "info mismatch at {}: {} vs {}",
            idx,
            actual,
            expected
        );
    }

    let expected_posterior = vec![
        vec![0.5 + inv_var * 1.0, 0.0],
        vec![0.0, 0.25 + inv_var * 4.0],
    ];
    let actual_posterior = dense_from_sparse(&posterior);
    assert_dense_close(&actual_posterior, &expected_posterior);
}

#[test]
fn precision_weighted_observations_match_manual_sparse_algebra() {
    let mut h_coo = CooMatrix::new(2, 2);
    h_coo.push(0, 0, 1.0);
    h_coo.push(1, 0, -1.0);
    h_coo.push(1, 1, 2.0);
    let h = SparseMatrix::from(&h_coo);

    let mut prior_coo = CooMatrix::new(2, 2);
    prior_coo.push(0, 0, 1.0);
    prior_coo.push(1, 1, 3.0);
    let prior = SparseMatrix::from(&prior_coo);

    let mut precision_coo = CooMatrix::new(2, 2);
    precision_coo.push(0, 0, 4.0);
    precision_coo.push(1, 1, 5.0);
    let precision = SparseMatrix::from(&precision_coo);

    let observations = Vector::from_vec(vec![2.0, -1.0]);
    let bias = Vector::from_vec(vec![0.5, 1.0]);

    let (posterior, info) = apply_gaussian_observations_with_precision(
        &prior,
        &h,
        &observations,
        Some(&bias),
        &precision,
    );

    let expected_info = [16.0, -20.0];
    for (actual, expected) in info.iter().zip(expected_info.iter()) {
        assert!((actual - expected).abs() < 1e-12);
    }

    let expected_update = vec![vec![9.0, -10.0], vec![-10.0, 20.0]];
    let actual_update = dense_from_sparse(&ht_precision_weighted_h(&h, &precision));
    assert_dense_close(&actual_update, &expected_update);

    let expected_posterior = vec![vec![10.0, -10.0], vec![-10.0, 23.0]];
    let actual_posterior = dense_from_sparse(&posterior);
    assert_dense_close(&actual_posterior, &expected_posterior);
}

#[test]
fn mixed_observation_terms_match_sequential_updates() {
    let mut prior_coo = CooMatrix::new(2, 2);
    prior_coo.push(0, 0, 1.0);
    prior_coo.push(1, 1, 2.0);
    let prior = SparseMatrix::from(&prior_coo);

    let mut scalar_h_coo = CooMatrix::new(1, 2);
    scalar_h_coo.push(0, 0, 2.0);
    let scalar_h = SparseMatrix::from(&scalar_h_coo);
    let scalar_y = Vector::from_vec(vec![3.0]);
    let scalar_bias = Vector::from_vec(vec![1.0]);

    let mut precision_h_coo = CooMatrix::new(2, 2);
    precision_h_coo.push(0, 0, 1.0);
    precision_h_coo.push(1, 1, -1.0);
    let precision_h = SparseMatrix::from(&precision_h_coo);
    let precision_y = Vector::from_vec(vec![0.5, -1.5]);
    let mut precision_coo = CooMatrix::new(2, 2);
    precision_coo.push(0, 0, 4.0);
    precision_coo.push(1, 1, 9.0);
    let precision = SparseMatrix::from(&precision_coo);

    let (after_scalar, scalar_info) =
        apply_gaussian_observations(&prior, &scalar_h, &scalar_y, Some(&scalar_bias), 0.5);
    let (expected_posterior, precision_info) = apply_gaussian_observations_with_precision(
        &after_scalar,
        &precision_h,
        &precision_y,
        None,
        &precision,
    );
    let expected_info = scalar_info + precision_info;

    let (actual_posterior, actual_info) = apply_linear_observation_terms(
        &prior,
        &[
            LinearObservationTerm::scalar_variance(&scalar_h, &scalar_y, Some(&scalar_bias), 0.5),
            LinearObservationTerm::precision(&precision_h, &precision_y, None, &precision),
        ],
    );

    assert_dense_close(
        &dense_from_sparse(&actual_posterior),
        &dense_from_sparse(&expected_posterior),
    );
    assert!((actual_info - expected_info).norm() < 1e-12);
}

#[test]
fn generic_observation_stack_builder_whitens_blocks() {
    let mut block_coo = CooMatrix::new(2, 2);
    block_coo.push(0, 0, 2.0);
    block_coo.push(1, 1, -4.0);
    let block = SparseMatrix::from(&block_coo);

    let mut builder = LinearObservationStackBuilder::new(4);
    builder
        .push_block(1, &block, &[3.0, -6.0], &[1.0, 2.0], 4.0)
        .expect("stacked observation term should assemble");
    let stacked = builder.finish();

    assert_eq!(stacked.observations.as_slice(), &[1.5, -3.0]);
    assert_eq!(stacked.bias.as_slice(), &[0.5, 1.0]);
    let dense = dense_from_sparse(&stacked.matrix);
    assert_dense_close(
        &dense,
        &[vec![0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, -2.0, 0.0]],
    );
}
