//! Honest resource records for sparse direct factorization and solves.
//!
//! Wall-clock timings are measured inside `gmrf-core`. Peak resident memory is
//! intentionally supplied by an external process monitor: portable in-process
//! RSS APIs generally report a process-wide high-water mark and cannot assign
//! memory to one solver phase without a fresh process per phase.

use std::time::Duration;

use crate::types::GmrfError;

/// Availability state for one resource quantity.
///
/// `NotRecorded` means that the quantity could be supplied or measured but was
/// not recorded by this operation. `Unavailable` means that the current
/// backend/API cannot expose the quantity without changing its meaning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResourceMeasurement<T> {
    /// A measured value.
    Measured(T),
    /// The operation did not record this otherwise meaningful quantity.
    NotRecorded,
    /// The quantity is unavailable for the stated reason.
    Unavailable(&'static str),
}

impl<T> ResourceMeasurement<T> {
    /// Return the measured value, if present.
    pub fn measured(&self) -> Option<&T> {
        match self {
            Self::Measured(value) => Some(value),
            Self::NotRecorded | Self::Unavailable(_) => None,
        }
    }

    /// Whether the backend/API explicitly declares this quantity unavailable.
    pub fn is_unavailable(&self) -> bool {
        matches!(self, Self::Unavailable(_))
    }
}

/// Resource usage for one non-repeated direct-solver phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectPhaseResourceUsage {
    /// Wall-clock duration of the phase.
    pub wall_time: ResourceMeasurement<Duration>,
    /// Peak resident-set size sampled externally while this phase ran.
    pub external_peak_rss_bytes: ResourceMeasurement<u64>,
}

impl DirectPhaseResourceUsage {
    fn measured_wall_time(wall_time: Duration) -> Self {
        Self {
            wall_time: ResourceMeasurement::Measured(wall_time),
            external_peak_rss_bytes: ResourceMeasurement::NotRecorded,
        }
    }

    fn not_recorded() -> Self {
        Self {
            wall_time: ResourceMeasurement::NotRecorded,
            external_peak_rss_bytes: ResourceMeasurement::NotRecorded,
        }
    }

    fn unavailable(reason: &'static str) -> Self {
        Self {
            wall_time: ResourceMeasurement::Unavailable(reason),
            external_peak_rss_bytes: ResourceMeasurement::NotRecorded,
        }
    }

    /// Attach a peak RSS value obtained by an external phase monitor.
    pub fn record_external_peak_rss(&mut self, bytes: u64) {
        self.external_peak_rss_bytes = ResourceMeasurement::Measured(bytes);
    }
}

/// Aggregated resource usage for repeated solves with one factorization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepeatedSolveResourceUsage {
    /// Sum of wall-clock durations for successful recorded solves.
    pub wall_time: ResourceMeasurement<Duration>,
    /// Peak resident-set size sampled externally over the recorded solves.
    pub external_peak_rss_bytes: ResourceMeasurement<u64>,
    /// Number of successful solve calls included in `wall_time`.
    pub calls: usize,
    /// Total number of right-hand-side columns included in `wall_time`.
    pub right_hand_sides: usize,
}

impl RepeatedSolveResourceUsage {
    fn empty() -> Self {
        Self {
            wall_time: ResourceMeasurement::NotRecorded,
            external_peak_rss_bytes: ResourceMeasurement::NotRecorded,
            calls: 0,
            right_hand_sides: 0,
        }
    }

    fn record(&mut self, elapsed: Duration, right_hand_sides: usize) {
        let total = match self.wall_time {
            ResourceMeasurement::Measured(current) => current.saturating_add(elapsed),
            ResourceMeasurement::NotRecorded | ResourceMeasurement::Unavailable(_) => elapsed,
        };
        self.wall_time = ResourceMeasurement::Measured(total);
        self.calls += 1;
        self.right_hand_sides += right_hand_sides;
    }

    /// Attach a peak RSS value obtained by an external monitor over the solves.
    pub fn record_external_peak_rss(&mut self, bytes: u64) {
        self.external_peak_rss_bytes = ResourceMeasurement::Measured(bytes);
    }
}

/// Sparse direct-solver pathway represented by a resource report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseDirectSolverKind {
    /// Symmetric positive-definite sparse Cholesky.
    Cholesky,
    /// General sparse LU with partial row pivoting.
    SparseLu,
}

/// Resource and storage report for one sparse direct factorization and its solves.
///
/// The four mathematical phases remain distinct. For sparse Cholesky,
/// `analysis` and `numeric_factorization` are measured separately. Faer's
/// high-level sparse LU API exposes one combined setup, so those two fields are
/// explicitly unavailable and `combined_factor_setup` is measured instead.
/// Matrix assembly is recorded only when `gmrf-core` performs it, as in KKT
/// construction; factorizing a caller-supplied matrix leaves it `NotRecorded`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseDirectResourceReport {
    /// Factorization/solve algorithm.
    pub solver: SparseDirectSolverKind,
    /// Coefficient-matrix dimension.
    pub dimension: usize,
    /// Stored coefficient-matrix nonzeros before factor fill.
    pub coefficient_nnz: usize,
    /// Numeric factor storage slots, where exposed by the backend.
    pub factor_storage_nonzeros: ResourceMeasurement<usize>,
    /// Coefficient-matrix assembly phase.
    pub assembly: DirectPhaseResourceUsage,
    /// Symbolic analysis/ordering phase.
    pub analysis: DirectPhaseResourceUsage,
    /// Numeric factorization after symbolic analysis.
    pub numeric_factorization: DirectPhaseResourceUsage,
    /// Combined symbolic and numeric setup when the backend cannot separate them.
    pub combined_factor_setup: DirectPhaseResourceUsage,
    /// Successful solves accumulated against this factorization.
    pub solve: RepeatedSolveResourceUsage,
}

impl SparseDirectResourceReport {
    pub(crate) fn cholesky(
        dimension: usize,
        coefficient_nnz: usize,
        factor_storage_nonzeros: usize,
        analysis_wall_time: Duration,
        factor_wall_time: Duration,
    ) -> Self {
        Self {
            solver: SparseDirectSolverKind::Cholesky,
            dimension,
            coefficient_nnz,
            factor_storage_nonzeros: ResourceMeasurement::Measured(factor_storage_nonzeros),
            assembly: DirectPhaseResourceUsage::not_recorded(),
            analysis: DirectPhaseResourceUsage::measured_wall_time(analysis_wall_time),
            numeric_factorization: DirectPhaseResourceUsage::measured_wall_time(factor_wall_time),
            combined_factor_setup: DirectPhaseResourceUsage::unavailable(
                "Cholesky analysis and numeric factorization are measured separately",
            ),
            solve: RepeatedSolveResourceUsage::empty(),
        }
    }

    pub(crate) fn sparse_lu(
        dimension: usize,
        coefficient_nnz: usize,
        combined_setup_wall_time: Duration,
    ) -> Self {
        const SEPARATE_LU_PHASE_REASON: &str =
            "the sparse LU backend does not expose separate symbolic and numeric phases";
        Self {
            solver: SparseDirectSolverKind::SparseLu,
            dimension,
            coefficient_nnz,
            factor_storage_nonzeros: ResourceMeasurement::Unavailable(
                "the sparse LU backend does not expose numeric factor storage",
            ),
            assembly: DirectPhaseResourceUsage::not_recorded(),
            analysis: DirectPhaseResourceUsage::unavailable(SEPARATE_LU_PHASE_REASON),
            numeric_factorization: DirectPhaseResourceUsage::unavailable(SEPARATE_LU_PHASE_REASON),
            combined_factor_setup: DirectPhaseResourceUsage::measured_wall_time(
                combined_setup_wall_time,
            ),
            solve: RepeatedSolveResourceUsage::empty(),
        }
    }

    pub(crate) fn record_assembly(&mut self, elapsed: Duration) {
        self.assembly = DirectPhaseResourceUsage::measured_wall_time(elapsed);
    }

    pub(crate) fn record_solve(
        &mut self,
        solver: SparseDirectSolverKind,
        dimension: usize,
        elapsed: Duration,
        right_hand_sides: usize,
    ) -> Result<(), GmrfError> {
        self.validate_for(solver, dimension)?;
        self.solve.record(elapsed, right_hand_sides);
        Ok(())
    }

    pub(crate) fn validate_for(
        &self,
        solver: SparseDirectSolverKind,
        dimension: usize,
    ) -> Result<(), GmrfError> {
        if self.solver != solver || self.dimension != dimension {
            return Err(GmrfError::DimensionMismatch(
                "resource report must match the solver kind and dimension",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_solve_usage_accumulates_calls_rhs_and_time() {
        let mut report = SparseDirectResourceReport::cholesky(
            3,
            7,
            5,
            Duration::from_millis(2),
            Duration::from_millis(3),
        );
        report
            .record_solve(
                SparseDirectSolverKind::Cholesky,
                3,
                Duration::from_millis(4),
                1,
            )
            .unwrap();
        report
            .record_solve(
                SparseDirectSolverKind::Cholesky,
                3,
                Duration::from_millis(6),
                8,
            )
            .unwrap();

        assert_eq!(report.solve.calls, 2);
        assert_eq!(report.solve.right_hand_sides, 9);
        assert_eq!(
            report.solve.wall_time,
            ResourceMeasurement::Measured(Duration::from_millis(10))
        );
    }

    #[test]
    fn resource_report_rejects_a_different_solver_or_dimension() {
        let mut report = SparseDirectResourceReport::sparse_lu(4, 10, Duration::ZERO);
        assert!(report
            .record_solve(SparseDirectSolverKind::Cholesky, 4, Duration::ZERO, 1)
            .is_err());
        assert!(report
            .record_solve(SparseDirectSolverKind::SparseLu, 3, Duration::ZERO, 1)
            .is_err());
    }
}
