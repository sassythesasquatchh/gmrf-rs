# Contributing

`gmrf-core` implements Gaussian precision algebra independently of any
particular mesh, finite element space, or application.

Contributions should:

- keep public sparse matrix and Gaussian model contracts explicit;
- preserve the coordinate and permutation conventions documented by each
  factorization type;
- state symmetry, positive-definiteness, rank, and noise assumptions;
- use the shared observation, constraint, covariance, sampling, and uncertainty
  implementations;
- expose unsupported cases as errors rather than silent numerical fallbacks;
- add focused unit tests and an integration test for cross-module behaviour.

Changes to sparse solvers or stochastic estimators should include a numerical
test that would detect an incorrect mean, covariance, or variance. Performance
work should report the matrix dimensions, nonzero counts, ordering, and solver
configuration needed to interpret the result.

Before submitting a change, run:

```text
cargo fmt --all --check
cargo check --release --all-targets --locked
cargo test --release --all-targets --locked
cargo clippy --release --all-targets --locked -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --release --no-deps --locked
```

The crate must remain independent of FEEC libraries and application-specific
model code.
