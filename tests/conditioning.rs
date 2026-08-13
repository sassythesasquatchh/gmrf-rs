use gmrf_core::observation::{apply_gaussian_observations, observation_selector};
use gmrf_core::solver::{DirectBackend, Solver, SolverAlgorithm, SolverConfig};
use gmrf_core::types::{CooMatrix, SparseMatrix, Vector};
use gmrf_core::{write_structured_points, Gmrf};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs::File;
use std::path::PathBuf;

fn isotropic_precision_grid(grid_size: usize, kappa: f64) -> SparseMatrix {
    let dimension = grid_size * grid_size;
    let mut coo = CooMatrix::new(dimension, dimension);
    let kappa2 = kappa * kappa;

    for y in 0..grid_size {
        for x in 0..grid_size {
            let idx = grid_index(grid_size, x, y);
            let mut degree = 0.0;
            if x > 0 {
                coo.push(idx, grid_index(grid_size, x - 1, y), -1.0);
                degree += 1.0;
            }
            if x + 1 < grid_size {
                coo.push(idx, grid_index(grid_size, x + 1, y), -1.0);
                degree += 1.0;
            }
            if y > 0 {
                coo.push(idx, grid_index(grid_size, x, y - 1), -1.0);
                degree += 1.0;
            }
            if y + 1 < grid_size {
                coo.push(idx, grid_index(grid_size, x, y + 1), -1.0);
                degree += 1.0;
            }
            coo.push(idx, idx, degree + kappa2);
        }
    }

    SparseMatrix::from(&coo)
}

fn grid_index(grid_size: usize, x: usize, y: usize) -> usize {
    y * grid_size + x
}

#[test]
fn posterior_samples_respect_high_certainty_observations() {
    let grid_size = 10;
    let kappa = 1.0;
    let noise_variance: f64 = 1e-6;
    let obs_points = [(2, 2), (6, 7), (8, 1)];
    let obs_values = vec![1.0, -0.5, 0.25];

    let dimension = grid_size * grid_size;
    let obs_indices: Vec<usize> = obs_points
        .iter()
        .map(|(x, y)| grid_index(grid_size, *x, *y))
        .collect();
    let observation_matrix = observation_selector(dimension, &obs_indices);
    let observations = Vector::from_vec(obs_values.clone());
    let prior_precision = isotropic_precision_grid(grid_size, kappa);

    let (posterior_precision, info) = apply_gaussian_observations(
        &prior_precision,
        &observation_matrix,
        &observations,
        None,
        noise_variance,
    );

    let solver_config = SolverConfig {
        algorithm: SolverAlgorithm::Direct(DirectBackend::SparseCholesky),
        ..Default::default()
    };
    let mut solver = Solver::new(solver_config);
    let posterior_mean = solver
        .solve_matrix(&posterior_precision, &info)
        .expect("posterior mean solve should succeed");
    let mut posterior = Gmrf::from_mean_and_precision(posterior_mean, posterior_precision)
        .expect("posterior build should succeed")
        .with_solver_config(solver_config);

    for (idx, obs) in obs_indices.iter().zip(obs_values.iter()) {
        let mean_val = posterior.mean()[*idx];
        assert!(
            (mean_val - obs).abs() < 1e-3,
            "posterior mean at obs index {} was {}, expected {}",
            idx,
            mean_val,
            obs
        );
    }

    let mut rng = StdRng::seed_from_u64(7);
    let threshold = 5.0 * noise_variance.sqrt();
    for _ in 0..5 {
        let sample = posterior.sample(&mut rng).expect("sample should succeed");
        for (idx, obs) in obs_indices.iter().zip(obs_values.iter()) {
            let diff = (sample[*idx] - obs).abs();
            assert!(
                diff < threshold,
                "sample at obs index {} deviated by {}, threshold {}",
                idx,
                diff,
                threshold
            );
        }
    }

    let sample_0 = posterior.sample(&mut rng).expect("sample should succeed");
    let sample_1 = posterior.sample(&mut rng).expect("sample should succeed");
    let sample_2 = posterior.sample(&mut rng).expect("sample should succeed");
    let fields = [
        ("mean", posterior.mean()),
        ("sample_0", &sample_0),
        ("sample_1", &sample_1),
        ("sample_2", &sample_2),
    ];
    let mut output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    output_path.push("posterior_sample.vtk");
    let mut file = File::create(&output_path).expect("create VTK output file");
    write_structured_points(&mut file, grid_size, &fields).expect("write VTK output");
}
