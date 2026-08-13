# Contributing

`gmrf-core` contains reusable Gaussian precision algebra and sparse solver
functionality. Downstream repositories provide FEEC assembly, application
models, and case-study workflows.

Preserve the supported sparse explicit-matrix APIs unless a coordinated
breaking change is explicitly planned. Reusable Gaussian algebra belongs here
rather than in downstream consumers. Numerical limitations, unsupported
cases, and solver assumptions must be documented and tested rather than hidden
behind silent fallbacks.

Before submitting a change, run:

```text
cargo fmt --all --check
cargo check --release --all-targets --locked
cargo test --release --all-targets --locked
cargo clippy --release --all-targets --locked -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --release --no-deps --locked
```

New features require focused unit tests. Cross-module behavior should also
receive an integration test under `tests/`. `gmrf-core` must remain independent
of FEEC libraries and application or integration crates.
