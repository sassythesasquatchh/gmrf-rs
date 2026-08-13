# Agent guidance

This repository contains the standalone `gmrf-core` crate.

## Ownership

`gmrf-core` owns generic Gaussian precision representations, sparse
factorizations and solves, linear Gaussian conditioning, equality constraints,
sampling, covariance actions, spatiotemporal precision storage, and uncertainty
estimators.

It must not depend on FEEC libraries or application/integration crates. FEEC
assembly, SPDE construction from FEEC operators, nonlinear model orchestration,
visualization, and case studies belong downstream.

## Development rules

- Preserve sparse explicit-matrix APIs used by FEEC–GMRF unless a coordinated
  breaking change is explicitly planned.
- Put reusable Gaussian algebra here rather than duplicating it in consumers.
- Add unit tests for every new feature and integration tests for new
  cross-module behavior.
- Run checks and tests in release mode before signing off.
- Surface numerical limitations, unsupported cases, and solver assumptions;
  do not hide them behind fallbacks.
