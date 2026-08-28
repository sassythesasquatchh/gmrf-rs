# gmrf-core

`gmrf-core` is a Rust library for Gaussian models represented by sparse
precision matrices. It provides the Gaussian algebra used by FEEC–GMRF, but it
does not depend on finite element or mesh types.

For a proper Gaussian vector

```text
x ~ N(mu, Q^-1),
```

the library stores the sparse precision \(Q\), the mean or information vector,
and reusable solver state. This representation is suited to Markov models in
which conditional independence makes \(Q\) sparse even when the covariance is
dense.

## Linear Gaussian conditioning

An observation term

```text
y = Hx + b + epsilon,    epsilon ~ N(0, R)
```

updates the precision and information vector by

```text
Q_post   = Q + H^T R^-1 H
eta_post = Q mu + H^T R^-1 (y - b).
```

`LinearObservationTerm` supports a common scalar variance or an explicit sparse
observation precision. `LinearObservationStackBuilder` combines multiple terms
without converting their row operators to dense matrices.

Exact constraints \(Cx=d\) use sparse KKT solves. Means, covariance actions,
samples, marginal variances, and transformed variances include the
constraint-induced covariance correction.

`SupportedGaussianSystem` covers the distinct case in which an ambient energy
is semidefinite and becomes a proper Gaussian law only after restriction to a
sparse affine support. It certifies full row rank of the support constraints
and positive energy on their nullspace without jitter, then uses an indefinite
KKT solve for means and covariance actions. The KKT matrix is not exposed as a
Gaussian precision. Sampling and support log determinants remain explicitly
unsupported until support-aware algorithms are implemented.

## Capabilities

- sparse precision matrices and information-form Gaussian models;
- fill-reducing sparse Cholesky analysis and factorization;
- reusable direct solves and configurable iterative solves;
- linear Gaussian observations with scalar or sparse noise precision;
- sparse row selection, stacking, composition, and transpose actions;
- exact linear equality constraints;
- certified sparse affine-support Gaussian laws with selected-coordinate
  covariance access;
- sampling from proper and constrained Gaussian models;
- latent and transformed covariance actions;
- exact marginal and transformed variances;
- Monte Carlo, Hutchinson, local Rao–Blackwellised, and selected-inverse
  uncertainty estimators;
- weighted covariance traces and estimator diagnostics;
- block-tridiagonal spatiotemporal precisions;
- Gaussian predictive diagnostics;
- optional VTK output for structured-grid examples.

Sparse matrix multiplication uses faer's sparse kernels when two assembled
operators are composed. Triplet builders are used for entry-wise assembly and
duplicate accumulation, where the output sparsity pattern is not already
known.

## Example

The included structured-grid example builds an isotropic sparse precision,
conditions it on observations, estimates posterior uncertainty, and writes a
VTK field:

```text
cargo run --release --example isotropic_grid_posterior
```

FEEC–GMRF supplies examples in which the precision and observation maps come
from differential-form finite elements.

## Use as a dependency

```toml
[dependencies]
gmrf-core = { git = "https://github.com/sassythesasquatchh/gmrf-rs", tag = "v0.1.0" }
```

## Building and testing

```text
cargo fmt --all --check
cargo check --release --all-targets --locked
cargo test --release --all-targets --locked
cargo clippy --release --all-targets --locked -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --release --no-deps --locked
```

## Acknowledgements and license

The library builds on Gaussian Markov random field code by Tim Weiland.
Patrick Dowd and contributors developed the Rust sparse-conditioning,
constraint, sampling, covariance, and uncertainty functionality described
above.

`gmrf-core` is distributed under the MIT license in `LICENSE`.
