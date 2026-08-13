//! Precision matrix abstractions and linear operator traits.
//!
//! The Julia package allows using explicit sparse matrices or matrix-free linear maps when
//! constructing a `GMRF`. This module mirrors that flexibility using the shared
//! `LinearOperator` trait and an enum to retain either a sparse matrix or boxed operator.

use crate::linear::LinearOperator;
use crate::types::{GmrfError, SparseMatrix, Vector};

/// A trait representing a matrix-free precision operator.
pub trait PrecisionOperator: LinearOperator {}

impl<T: LinearOperator + ?Sized> PrecisionOperator for T {}

/// Storage for either a concrete precision matrix or a matrix-free operator.
pub enum PrecisionStorage {
    /// Concrete sparse precision matrix (preferred for sampling and solves).
    Matrix(SparseMatrix),
    /// Matrix-free linear operator with known dimension.
    Operator(Box<dyn PrecisionOperator>),
}

impl PrecisionStorage {
    /// Returns the number of rows/columns represented by the precision.
    pub fn dimension(&self) -> usize {
        match self {
            PrecisionStorage::Matrix(matrix) => matrix.nrows(),
            PrecisionStorage::Operator(operator) => operator.dimension(),
        }
    }

    /// Access the underlying matrix reference when available.
    pub fn as_matrix(&self) -> Option<&SparseMatrix> {
        match self {
            PrecisionStorage::Matrix(matrix) => Some(matrix),
            PrecisionStorage::Operator { .. } => None,
        }
    }

    /// Apply the precision regardless of whether it is matrix-based or matrix-free.
    pub fn apply(&self, x: &Vector) -> Result<Vector, GmrfError> {
        match self {
            PrecisionStorage::Matrix(matrix) => {
                if x.len() != matrix.ncols() {
                    return Err(GmrfError::DimensionMismatch(
                        "input length must match precision dimension",
                    ));
                }
                Ok(matrix * x)
            }
            PrecisionStorage::Operator(operator) => operator.apply(x),
        }
    }
}
