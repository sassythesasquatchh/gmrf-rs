# Contributing

`gmrf-core` contains reusable Gaussian precision algebra and sparse solver
functionality. Downstream repositories provide FEEC assembly, application
models, and case-study workflows.

Before submitting a change, run:

```text
cargo fmt --all --check
cargo check --release --all-targets --locked
cargo test --release --all-targets --locked
cargo clippy --release --all-targets --locked -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --release --no-deps --locked
```

New features require focused unit tests. Cross-module behavior should also
receive an integration test under `tests/`.
