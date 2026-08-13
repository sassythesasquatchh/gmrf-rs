//! Linear operator abstractions mirroring the Julia `LinearMaps` usage.
//!
//! These traits allow the core crate to work with both explicit sparse matrices and
//! matrix-free operators. They are intentionally lightweight so they can back
//! precision operators, preconditioners, and composed transforms without forcing
//! a particular backend.

use crate::types::{CooMatrix, DenseMatrix, GmrfError, Permutation, SparseMatrix, Vector};

/// A trait representing a generic linear operator `y = A * x`.
pub trait LinearOperator: Send + Sync {
    /// Dimension of the operator.
    fn dimension(&self) -> usize;

    /// Apply the operator to a vector, returning the result.
    fn apply(&self, x: &Vector) -> Result<Vector, GmrfError>;
}

/// A concrete operator backed by a sparse matrix.
pub struct MatrixOperator {
    matrix: SparseMatrix,
}

impl MatrixOperator {
    /// Create a linear operator from a sparse matrix.
    pub fn new(matrix: SparseMatrix) -> Self {
        Self { matrix }
    }
}

impl LinearOperator for MatrixOperator {
    fn dimension(&self) -> usize {
        self.matrix.nrows()
    }

    fn apply(&self, x: &Vector) -> Result<Vector, GmrfError> {
        if x.len() != self.matrix.ncols() {
            return Err(GmrfError::DimensionMismatch(
                "input length must match matrix column dimension",
            ));
        }

        Ok(self.matrix.mul_vec(x))
    }
}

/// Compose two operators B(A(x)) to allow lightweight chaining.
pub struct ComposedOperator<A: LinearOperator, B: LinearOperator> {
    first: A,
    second: B,
}

impl<A: LinearOperator, B: LinearOperator> ComposedOperator<A, B> {
    /// Build a composed operator `second(first(x))`.
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A: LinearOperator, B: LinearOperator> LinearOperator for ComposedOperator<A, B> {
    fn dimension(&self) -> usize {
        self.first.dimension()
    }

    fn apply(&self, x: &Vector) -> Result<Vector, GmrfError> {
        let intermediate = self.first.apply(x)?;
        self.second.apply(&intermediate)
    }
}

/// Sparse row-wise operator for rectangular transforms `y = A x`.
///
/// This is intentionally lightweight and backend-agnostic. It is primarily used for
/// transformed variance calculations where each output row is a sparse linear functional
/// over the latent state.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseRowOperator {
    pub ncols: usize,
    pub rows: Vec<Vec<(usize, f64)>>,
}

impl SparseRowOperator {
    /// Build an operator from explicit sparse rows.
    pub fn new(ncols: usize, rows: Vec<Vec<(usize, f64)>>) -> Result<Self, GmrfError> {
        for row in &rows {
            for (col, value) in row {
                if *col >= ncols {
                    return Err(GmrfError::DimensionMismatch(
                        "sparse row operator column exceeds input dimension",
                    ));
                }
                if !value.is_finite() {
                    return Err(GmrfError::NumericalInstability(
                        "sparse row operator contains non-finite entries",
                    ));
                }
            }
        }

        Ok(Self { ncols, rows })
    }

    /// Build an operator from a sparse matrix, preserving the row structure.
    pub fn from_sparse_matrix(matrix: &SparseMatrix) -> Result<Self, GmrfError> {
        let mut rows = vec![Vec::new(); matrix.nrows()];
        for (row, col, value) in matrix.triplet_iter() {
            if *value != 0.0 {
                rows[row].push((col, *value));
            }
        }
        Self::new(matrix.ncols(), rows)
    }

    /// Convert the row operator into an explicit sparse matrix.
    pub fn to_sparse_matrix(&self) -> SparseMatrix {
        let mut coo = CooMatrix::new(self.nrows(), self.ncols);
        for (row, entries) in self.rows.iter().enumerate() {
            for (col, value) in entries {
                if *value != 0.0 {
                    coo.push(row, *col, *value);
                }
            }
        }
        SparseMatrix::from(&coo)
    }

    /// Build an operator from a dense matrix, dropping entries with magnitude `<= drop_tolerance`.
    pub fn from_dense_matrix(matrix: &DenseMatrix, drop_tolerance: f64) -> Result<Self, GmrfError> {
        if !drop_tolerance.is_finite() {
            return Err(GmrfError::NumericalInstability(
                "dense-to-row conversion drop tolerance must be finite",
            ));
        }

        let tol = drop_tolerance.abs();
        let mut rows = Vec::with_capacity(matrix.nrows());
        for row in 0..matrix.nrows() {
            let mut entries = Vec::new();
            for col in 0..matrix.ncols() {
                let value = matrix[(row, col)];
                if value.abs() > tol {
                    entries.push((col, value));
                }
            }
            rows.push(entries);
        }
        Self::new(matrix.ncols(), rows)
    }

    /// Identity operator on `size` coordinates.
    pub fn identity(size: usize) -> Self {
        Self {
            ncols: size,
            rows: (0..size).map(|i| vec![(i, 1.0)]).collect(),
        }
    }

    /// Number of output rows.
    pub fn nrows(&self) -> usize {
        self.rows.len()
    }

    /// Select output rows, preserving their order in `row_indices`.
    pub fn select_rows(&self, row_indices: &[usize]) -> Result<Self, GmrfError> {
        let mut rows = Vec::with_capacity(row_indices.len());
        for row in row_indices {
            let entries = self.rows.get(*row).ok_or(GmrfError::DimensionMismatch(
                "selected sparse row operator row is out of bounds",
            ))?;
            rows.push(entries.clone());
        }
        Self::new(self.ncols, rows)
    }

    /// Expand one sparse output row into a dense latent-space vector.
    pub fn row_as_vector(&self, row_index: usize) -> Result<Vector, GmrfError> {
        let row = self
            .rows
            .get(row_index)
            .ok_or(GmrfError::DimensionMismatch(
                "operator row index is out of bounds",
            ))?;
        let mut out = Vector::zeros(self.ncols);
        for (col, value) in row {
            out[*col] += *value;
        }
        Ok(out)
    }

    /// Apply the operator.
    pub fn apply(&self, input: &Vector) -> Result<Vector, GmrfError> {
        if input.len() != self.ncols {
            return Err(GmrfError::DimensionMismatch(
                "operator input length must match column dimension",
            ));
        }

        Ok(Vector::from_iterator(
            self.nrows(),
            self.rows.iter().map(|row| {
                row.iter()
                    .map(|(col, value)| *value * input[*col])
                    .sum::<f64>()
            }),
        ))
    }

    /// Apply the transpose operator.
    pub fn apply_transpose(&self, input: &Vector) -> Result<Vector, GmrfError> {
        if input.len() != self.nrows() {
            return Err(GmrfError::DimensionMismatch(
                "transpose input length must match row dimension",
            ));
        }

        let mut out = Vector::zeros(self.ncols);
        for (row_index, row) in self.rows.iter().enumerate() {
            let weight = input[row_index];
            if weight == 0.0 {
                continue;
            }
            for (col, value) in row {
                out[*col] += weight * *value;
            }
        }
        Ok(out)
    }

    /// Return the operator with columns mapped from original coordinates to
    /// Cholesky/permuted coordinates, i.e. `A_tilde = A P^T`.
    pub fn permute_columns_to_factor(&self, permutation: &Permutation) -> Result<Self, GmrfError> {
        if permutation.dimension() != self.ncols {
            return Err(GmrfError::DimensionMismatch(
                "permutation dimension must match operator column count",
            ));
        }
        let rows = self
            .rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|(col, value)| (permutation.orig_to_perm[*col], *value))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Self::new(self.ncols, rows)
    }

    /// Stack operators vertically.
    pub fn stack(operators: &[&SparseRowOperator]) -> Result<Self, GmrfError> {
        let Some(first) = operators.first() else {
            return Err(GmrfError::DimensionMismatch(
                "at least one operator is required for stacking",
            ));
        };
        let ncols = first.ncols;
        if operators.iter().any(|operator| operator.ncols != ncols) {
            return Err(GmrfError::DimensionMismatch(
                "all stacked operators must have the same column dimension",
            ));
        }

        let mut rows = Vec::new();
        for operator in operators {
            rows.extend(operator.rows.iter().cloned());
        }
        Ok(Self { ncols, rows })
    }

    /// Compose two operators `left(right(x))`.
    pub fn compose(left: &SparseRowOperator, right: &SparseRowOperator) -> Result<Self, GmrfError> {
        if left.ncols != right.nrows() {
            return Err(GmrfError::DimensionMismatch(
                "operator dimensions are incompatible for composition",
            ));
        }

        let mut rows = Vec::with_capacity(left.nrows());
        for left_row in &left.rows {
            let mut combined = std::collections::BTreeMap::<usize, f64>::new();
            for (intermediate, weight) in left_row {
                for (col, value) in &right.rows[*intermediate] {
                    *combined.entry(*col).or_insert(0.0) += *weight * *value;
                }
            }
            rows.push(
                combined
                    .into_iter()
                    .filter_map(|(col, value)| (value != 0.0).then_some((col, value)))
                    .collect(),
            );
        }

        Ok(Self {
            ncols: right.ncols,
            rows,
        })
    }
}

/// Linear operator paired with a known square root operator, mirroring Julia's `LinearMapWithSqrt`.
pub struct OperatorWithSqrt<O: LinearOperator, S: LinearOperator> {
    operator: O,
    sqrt: S,
}

impl<O: LinearOperator, S: LinearOperator> OperatorWithSqrt<O, S> {
    /// Create a new operator-with-sqrt, verifying dimensions match.
    pub fn new(operator: O, sqrt: S) -> Result<Self, GmrfError> {
        if operator.dimension() != sqrt.dimension() {
            return Err(GmrfError::DimensionMismatch(
                "operator and square root must share dimensions",
            ));
        }
        Ok(Self { operator, sqrt })
    }

    /// Access the stored square root operator.
    pub fn sqrt(&self) -> &S {
        &self.sqrt
    }
}

impl<O: LinearOperator, S: LinearOperator> LinearOperator for OperatorWithSqrt<O, S> {
    fn dimension(&self) -> usize {
        self.operator.dimension()
    }

    fn apply(&self, x: &Vector) -> Result<Vector, GmrfError> {
        self.operator.apply(x)
    }
}

/// Build a Kronecker product of two matrix-backed operators.
pub fn kronecker(a: &MatrixOperator, b: &MatrixOperator) -> MatrixOperator {
    let (a_rows, a_cols) = (a.matrix.nrows(), a.matrix.ncols());
    let (b_rows, b_cols) = (b.matrix.nrows(), b.matrix.ncols());
    let mut coo = CooMatrix::new(a_rows * b_rows, a_cols * b_cols);

    for (ai, aj, av) in a.matrix.triplet_iter() {
        for (bi, bj, bv) in b.matrix.triplet_iter() {
            let row = ai * b_rows + bi;
            let col = aj * b_cols + bj;
            coo.push(row, col, *av * *bv);
        }
    }

    MatrixOperator::new(SparseMatrix::from(&coo))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_matrix(size: usize) -> SparseMatrix {
        let mut coo = CooMatrix::new(size, size);
        for i in 0..size {
            coo.push(i, i, 1.0);
        }
        SparseMatrix::from(&coo)
    }

    #[test]
    fn operator_with_sqrt_applies_operator() {
        let mat = identity_matrix(3);
        let op = MatrixOperator::new(mat.clone());
        let sqrt = MatrixOperator::new(mat);
        let op_with_sqrt = OperatorWithSqrt::new(op, sqrt).unwrap();
        let x = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let y = op_with_sqrt.apply(&x).unwrap();
        assert_eq!(y, x);
        let y_sqrt = op_with_sqrt.sqrt().apply(&x).unwrap();
        assert_eq!(y_sqrt, x);
    }

    #[test]
    fn sparse_row_operator_selects_rows_and_expands_to_matrix() {
        let operator = SparseRowOperator::new(
            4,
            vec![vec![(0, 1.0), (2, -1.0)], vec![(1, 2.0)], vec![(3, 4.0)]],
        )
        .unwrap();
        let selected = operator.select_rows(&[2, 0]).unwrap();
        assert_eq!(
            selected
                .apply(&Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0]))
                .unwrap(),
            Vector::from_vec(vec![16.0, -2.0])
        );
        let matrix = selected.to_sparse_matrix();
        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 4);
        assert_eq!(matrix.nnz(), 3);
    }

    #[test]
    fn kronecker_builds_correct_dimension() {
        let a = MatrixOperator::new(identity_matrix(2));
        let b = MatrixOperator::new(identity_matrix(3));
        let kron = kronecker(&a, &b);
        assert_eq!(kron.dimension(), 6);
        let v = Vector::from_vec(vec![1.0; 6]);
        let out = kron.apply(&v).unwrap();
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn sparse_row_operator_stacks_and_applies() {
        let left = SparseRowOperator::new(3, vec![vec![(0, 1.0), (2, -1.0)]]).unwrap();
        let right = SparseRowOperator::new(3, vec![vec![(1, 2.0)]]).unwrap();
        let stacked = SparseRowOperator::stack(&[&left, &right]).unwrap();

        let input = Vector::from_vec(vec![3.0, 4.0, 1.0]);
        let output = stacked.apply(&input).unwrap();

        assert_eq!(output.len(), 2);
        assert!((output[0] - 2.0).abs() < 1e-12);
        assert!((output[1] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn sparse_row_operator_compose_matches_manual() {
        let right =
            SparseRowOperator::new(3, vec![vec![(0, 1.0), (1, 2.0)], vec![(1, -1.0), (2, 0.5)]])
                .unwrap();
        let left = SparseRowOperator::new(2, vec![vec![(0, 2.0)], vec![(1, -3.0)]]).unwrap();

        let composed = SparseRowOperator::compose(&left, &right).unwrap();
        let input = Vector::from_vec(vec![1.0, -2.0, 4.0]);
        let manual = left.apply(&right.apply(&input).unwrap()).unwrap();
        let actual = composed.apply(&input).unwrap();

        assert!((manual - actual).norm() < 1e-12);
    }

    #[test]
    fn sparse_row_operator_apply_transpose_matches_manual() {
        let operator =
            SparseRowOperator::new(3, vec![vec![(0, 1.0), (2, -2.0)], vec![(1, 0.5), (2, 3.0)]])
                .unwrap();
        let weights = Vector::from_vec(vec![2.0, -1.0]);
        let applied = operator.apply_transpose(&weights).unwrap();

        assert!((applied[0] - 2.0).abs() < 1e-12);
        assert!((applied[1] + 0.5).abs() < 1e-12);
        assert!((applied[2] + 7.0).abs() < 1e-12);
    }
}
