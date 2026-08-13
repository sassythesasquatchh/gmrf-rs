//! Example: isotropic 2D Gaussian prior on a grid, conditioning on observations, and sampling.
//!
//! This demonstrates a sparse-only workflow for building a simple isotropic precision
//! (2D Laplacian + κ² I), conditioning on noisy point observations, sampling from the
//! posterior, and writing results to a CSV for visualization.

use gmrf_core::observation::{apply_gaussian_observations, observation_selector};
use gmrf_core::solver::{DirectBackend, Solver, SolverAlgorithm, SolverConfig};
use gmrf_core::types::{CooMatrix, SparseMatrix, Vector};
use gmrf_core::{write_structured_points, Gmrf};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grid_size = 20;
    let kappa = 1.0;
    let noise_variance: f64 = 0.0000001;
    let num_samples = 3;
    let solver_config = SolverConfig {
        algorithm: SolverAlgorithm::Direct(DirectBackend::SparseCholesky),
        ..Default::default()
    };

    let prior_precision = isotropic_precision_grid(grid_size, kappa);
    let dimension = grid_size * grid_size;

    // Build a sparse observation matrix selecting a handful of grid points.
    let obs_points = [(2, 2), (5, 15), (10, 3), (15, 12), (18, 6)];
    let obs_indices: Vec<usize> = obs_points
        .iter()
        .map(|(x, y)| grid_index(grid_size, *x, *y))
        .collect();
    let observation_matrix = observation_selector(dimension, &obs_indices);

    // Generate synthetic data from the prior.
    let mut rng = StdRng::seed_from_u64(2024);
    let mut prior =
        Gmrf::from_mean_and_precision(Vector::zeros(dimension), prior_precision.clone())?
            .with_solver_config(solver_config);
    let latent_true = prior.sample(&mut rng)?;

    let noise = Normal::new(0.0, noise_variance.sqrt())?;
    let mut observations = &observation_matrix * &latent_true;
    for i in 0..observations.len() {
        observations[i] += noise.sample(&mut rng);
    }

    // Condition on observations: Q_post = Q + Hᵀ R⁻¹ H, η = Hᵀ R⁻¹ y.
    let (posterior_precision, info) = apply_gaussian_observations(
        &prior_precision,
        &observation_matrix,
        &observations,
        None,
        noise_variance,
    );

    let mut solver = Solver::new(solver_config);
    let posterior_mean = solver.solve_matrix(&posterior_precision, &info)?;
    let mut posterior = Gmrf::from_mean_and_precision(posterior_mean, posterior_precision)?
        .with_solver_config(solver_config);

    let output_path = "isotropic_posterior_samples.vtk";
    write_vtk(
        output_path,
        grid_size,
        &mut posterior,
        num_samples,
        &mut rng,
    )?;
    println!("Wrote samples to {}", output_path);
    Ok(())
}

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

fn write_vtk(
    path: &str,
    grid_size: usize,
    posterior: &mut Gmrf,
    num_samples: usize,
    rng: &mut impl Rng,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let mut fields: Vec<(String, Vector)> = Vec::with_capacity(num_samples + 1);
    fields.push(("mean".to_string(), posterior.mean().clone()));
    for sample_idx in 0..num_samples {
        let draw = posterior.sample(rng)?;
        fields.push((format!("sample_{}", sample_idx), draw));
    }

    let field_refs: Vec<(&str, &Vector)> = fields
        .iter()
        .map(|(name, values)| (name.as_str(), values))
        .collect();
    write_structured_points(&mut writer, grid_size, &field_refs)?;
    writer.flush()?;
    Ok(())
}
