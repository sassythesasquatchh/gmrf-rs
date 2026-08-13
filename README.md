# GMRF (Rust)

`gmrf-core` provides the Gaussian and sparse-linear-algebra layer used by
FEEC–GMRF. It owns sparse precision representations, factorizations and
solves, linear Gaussian conditioning, equality constraints, sampling,
spatiotemporal precision storage, covariance actions, and uncertainty
estimators.

The initial `0.1.0` release deliberately contains one crate. Finite-element
assembly, SPDE model construction, nonlinear model orchestration,
visualization, and application-specific workflows belong in downstream
libraries rather than this repository.

## Use from Git

```toml
[dependencies]
gmrf-core = { git = "https://github.com/sassythesasquatchh/gmrf-rs", tag = "v0.1.0" }
```

## Verification

```text
cargo fmt --all --check
cargo check --release --all-targets --locked
cargo test --release --all-targets --locked
cargo clippy --release --all-targets --locked -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --release --no-deps --locked
```

## Lineage

This repository is a curated, clean-history import of the supported Rust core
from an earlier GaussianMarkovRandomFields.jl port. See [UPSTREAM.md](UPSTREAM.md)
for the exact source commit and attribution. The code is distributed under the
MIT license in [LICENSE](LICENSE).
