//! Criterion benchmarks for core solver routines.
//! These track solve and sampling performance to catch regressions
//! as solver implementations evolve.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use gmrf_core::types::{CooMatrix, SparseMatrix};
use gmrf_core::{Gmrf, Solver, Vector};
use rand::{rngs::StdRng, SeedableRng};

fn identity_precision(size: usize, scale: f64) -> SparseMatrix {
    let mut coo = CooMatrix::new(size, size);
    for i in 0..size {
        coo.push(i, i, scale);
    }
    SparseMatrix::from(&coo)
}

fn bench_precision_solve(c: &mut Criterion) {
    let precision = identity_precision(32, 2.0);
    let rhs = Vector::from_element(32, 1.0);
    let mut solver = Solver::default();

    c.bench_function("solve_precision_32", |b| {
        b.iter(|| {
            solver
                .solve_matrix(&precision, &rhs)
                .expect("solve succeeds")
        });
    });
}

fn bench_gmrf_sampling(c: &mut Criterion) {
    let precision = identity_precision(16, 1.0);
    let mean = Vector::zeros(16);
    let mut gmrf = Gmrf::from_mean_and_precision(mean, precision).expect("gmrf build");

    c.bench_function("gmrf_sample_16", |b| {
        b.iter_batched(
            || StdRng::seed_from_u64(42),
            |mut rng| gmrf.sample(&mut rng).expect("sample"),
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(solver_benches, bench_precision_solve, bench_gmrf_sampling);
criterion_main!(solver_benches);
