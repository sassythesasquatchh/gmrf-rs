//! Structured spacetime precision and observation builders.

use crate::observation::{add_sparse, LinearObservationStackBuilder, StackedObservationSystem};
use crate::types::{CooMatrix, GmrfError, SparseMatrix, Vector};

#[derive(Debug, Clone)]
pub struct BlockTridiagonalPrecision {
    block_size: usize,
    diagonal_blocks: Vec<SparseMatrix>,
    lower_blocks: Vec<SparseMatrix>,
}

impl BlockTridiagonalPrecision {
    pub fn new(
        diagonal_blocks: Vec<SparseMatrix>,
        lower_blocks: Vec<SparseMatrix>,
    ) -> Result<Self, GmrfError> {
        let Some(first) = diagonal_blocks.first() else {
            return Err(GmrfError::DimensionMismatch(
                "block-tridiagonal precision requires at least one diagonal block",
            ));
        };
        if first.nrows() != first.ncols() {
            return Err(GmrfError::DimensionMismatch(
                "diagonal blocks must be square",
            ));
        }
        let block_size = first.nrows();
        if diagonal_blocks
            .iter()
            .any(|block| block.nrows() != block_size || block.ncols() != block_size)
        {
            return Err(GmrfError::DimensionMismatch(
                "all diagonal blocks must share the same square dimension",
            ));
        }
        if lower_blocks.len() + 1 != diagonal_blocks.len() {
            return Err(GmrfError::DimensionMismatch(
                "lower block count must be exactly one less than the diagonal block count",
            ));
        }
        if lower_blocks
            .iter()
            .any(|block| block.nrows() != block_size || block.ncols() != block_size)
        {
            return Err(GmrfError::DimensionMismatch(
                "all off-diagonal blocks must match the diagonal block size",
            ));
        }

        Ok(Self {
            block_size,
            diagonal_blocks,
            lower_blocks,
        })
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn block_count(&self) -> usize {
        self.diagonal_blocks.len()
    }

    pub fn dimension(&self) -> usize {
        self.block_size * self.block_count()
    }

    pub fn diagonal_blocks(&self) -> &[SparseMatrix] {
        &self.diagonal_blocks
    }

    pub fn lower_blocks(&self) -> &[SparseMatrix] {
        &self.lower_blocks
    }

    pub fn to_sparse(&self) -> SparseMatrix {
        let dimension = self.dimension();
        let mut coo = CooMatrix::new(dimension, dimension);
        for (block_index, block) in self.diagonal_blocks.iter().enumerate() {
            let offset = block_index * self.block_size;
            for (row, col, value) in block.triplet_iter() {
                coo.push(offset + row, offset + col, *value);
            }
        }
        for (block_index, block) in self.lower_blocks.iter().enumerate() {
            let row_offset = (block_index + 1) * self.block_size;
            let col_offset = block_index * self.block_size;
            for (row, col, value) in block.triplet_iter() {
                coo.push(row_offset + row, col_offset + col, *value);
                coo.push(col_offset + col, row_offset + row, *value);
            }
        }
        SparseMatrix::from(&coo)
    }

    pub fn cholesky_sqrt_lower(&self) -> Result<crate::SparseCholeskyFactor, GmrfError> {
        self.to_sparse().cholesky_sqrt_lower()
    }

    pub fn solve(&self, rhs: &Vector) -> Result<Vector, GmrfError> {
        let precision = self.to_sparse();
        let mut solver = crate::Solver::default();
        solver.solve_matrix(&precision, rhs)
    }
}

#[derive(Debug, Clone)]
pub struct TimeStackedObservationBuilder {
    slice_count: usize,
    slice_dimension: usize,
    inner: LinearObservationStackBuilder,
}

impl TimeStackedObservationBuilder {
    pub fn new(slice_count: usize, slice_dimension: usize) -> Self {
        Self {
            slice_count,
            slice_dimension,
            inner: LinearObservationStackBuilder::new(slice_count * slice_dimension),
        }
    }

    pub fn push_slice_block(
        &mut self,
        slice_index: usize,
        block: &SparseMatrix,
        observations: &[f64],
        bias: &[f64],
        variance: f64,
    ) -> Result<(), GmrfError> {
        self.validate_slice_block(slice_index, block, observations, bias, variance)?;
        let col_offset = slice_index * self.slice_dimension;
        self.inner
            .push_block(col_offset, block, observations, bias, variance)
    }

    pub fn push_transition_block(
        &mut self,
        left_slice: usize,
        left_block: &SparseMatrix,
        right_block: &SparseMatrix,
        observations: &[f64],
        bias: &[f64],
        variance: f64,
    ) -> Result<(), GmrfError> {
        if left_slice + 1 >= self.slice_count {
            return Err(GmrfError::DimensionMismatch(
                "transition block must fit inside the time grid",
            ));
        }
        self.validate_block_rows(left_block, observations, bias, variance)?;
        if right_block.nrows() != left_block.nrows() || right_block.ncols() != self.slice_dimension
        {
            return Err(GmrfError::DimensionMismatch(
                "right transition block must match the left block row count and slice dimension",
            ));
        }
        let left_offset = left_slice * self.slice_dimension;
        let right_offset = (left_slice + 1) * self.slice_dimension;
        self.inner.push_blocks(
            &[(left_offset, left_block), (right_offset, right_block)],
            observations,
            bias,
            variance,
        )
    }

    pub fn finish(self) -> StackedObservationSystem {
        self.inner.finish()
    }

    fn validate_slice_block(
        &self,
        slice_index: usize,
        block: &SparseMatrix,
        observations: &[f64],
        bias: &[f64],
        variance: f64,
    ) -> Result<(), GmrfError> {
        if slice_index >= self.slice_count {
            return Err(GmrfError::DimensionMismatch(
                "slice block must fit inside the time grid",
            ));
        }
        self.validate_block_rows(block, observations, bias, variance)?;
        if block.ncols() != self.slice_dimension {
            return Err(GmrfError::DimensionMismatch(
                "slice block column count must match the slice dimension",
            ));
        }
        Ok(())
    }

    fn validate_block_rows(
        &self,
        block: &SparseMatrix,
        observations: &[f64],
        bias: &[f64],
        variance: f64,
    ) -> Result<(), GmrfError> {
        if block.nrows() != observations.len() || block.nrows() != bias.len() {
            return Err(GmrfError::DimensionMismatch(
                "observation rows, observations, and bias lengths must match",
            ));
        }
        if !variance.is_finite() || variance <= 0.0 {
            return Err(GmrfError::DimensionMismatch(
                "observation variance must be finite and positive",
            ));
        }
        Ok(())
    }
}

pub fn add_sparse_blocks(lhs: &SparseMatrix, rhs: &SparseMatrix) -> SparseMatrix {
    add_sparse(lhs, rhs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sparse(nrows: usize, ncols: usize, entries: &[(usize, usize, f64)]) -> SparseMatrix {
        let mut coo = CooMatrix::new(nrows, ncols);
        for &(row, col, value) in entries {
            coo.push(row, col, value);
        }
        SparseMatrix::from(&coo)
    }

    fn dense_from_sparse(matrix: &SparseMatrix) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; matrix.ncols()]; matrix.nrows()];
        for (row, col, value) in matrix.triplet_iter() {
            dense[row][col] += *value;
        }
        dense
    }

    #[test]
    fn block_tridiagonal_materializes_expected_sparse_matrix() {
        let diag0 = sparse(2, 2, &[(0, 0, 2.0), (1, 1, 3.0)]);
        let diag1 = sparse(2, 2, &[(0, 0, 5.0), (1, 1, 7.0)]);
        let lower = sparse(2, 2, &[(0, 1, -1.5)]);
        let precision = BlockTridiagonalPrecision::new(vec![diag0, diag1], vec![lower]).unwrap();
        let dense = dense_from_sparse(&precision.to_sparse());
        assert_eq!(
            dense,
            vec![
                vec![2.0, 0.0, 0.0, 0.0],
                vec![0.0, 3.0, -1.5, 0.0],
                vec![0.0, -1.5, 5.0, 0.0],
                vec![0.0, 0.0, 0.0, 7.0],
            ]
        );
    }

    #[test]
    fn stacked_observation_builder_scales_rows_by_variance() {
        let block = sparse(2, 2, &[(0, 0, 1.0), (1, 1, 2.0)]);
        let mut builder = TimeStackedObservationBuilder::new(2, 2);
        builder
            .push_slice_block(1, &block, &[3.0, 4.0], &[1.0, -1.0], 4.0)
            .unwrap();
        let system = builder.finish();
        let dense = dense_from_sparse(&system.matrix);
        assert_eq!(
            dense,
            vec![vec![0.0, 0.0, 0.5, 0.0], vec![0.0, 0.0, 0.0, 1.0]]
        );
        assert_eq!(system.observations.as_slice(), &[1.5, 2.0]);
        assert_eq!(system.bias.as_slice(), &[0.5, -0.5]);
        assert_eq!(system.noise_variance, 1.0);
    }

    #[test]
    fn linear_observation_stack_builder_places_blocks_at_offsets() {
        let block = sparse(2, 2, &[(0, 0, 1.0), (1, 1, -2.0)]);
        let mut builder = LinearObservationStackBuilder::new(5);
        builder
            .push_blocks(&[(0, &block), (3, &block)], &[2.0, 4.0], &[1.0, -1.0], 4.0)
            .expect("stacked observation term should assemble");
        let system = builder.finish();
        assert_eq!(system.matrix.nrows(), 2);
        assert_eq!(system.matrix.ncols(), 5);
        assert_eq!(system.observations.as_slice(), &[1.0, 2.0]);
        assert_eq!(system.bias.as_slice(), &[0.5, -0.5]);
        let dense = dense_from_sparse(&system.matrix);
        assert_eq!(dense[0], vec![0.5, 0.0, 0.0, 0.5, 0.0]);
        assert_eq!(dense[1], vec![0.0, -1.0, 0.0, 0.0, -1.0]);
    }
}
