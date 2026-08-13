//! Shared math aliases and errors for Gaussian Markov Random Field primitives.
//!
//! The gmrf workspace uses faer + faer-sparse for linear algebra. This module provides
//! thin wrappers and helpers so downstream crates don't depend on faer directly.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::solvers::Solve;
use faer::perm::PermRef;
use faer::sparse::linalg::cholesky::supernodal::SupernodalLltRef;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, LltRef, SymbolicCholesky,
    SymbolicCholeskyRaw, SymmetricOrdering,
};
use faer::sparse::linalg::solvers::Lu as FaerSparseLu;
use faer::sparse::{SparseColMat, SparseColMatRef, Triplet};
use faer::Mat;
use faer::{get_global_parallelism, Conj, Side, Unbind};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Sub, SubAssign},
    sync::Arc,
};
use thiserror::Error;

/// Dense matrix alias for design matrices and Jacobians.
pub type DenseMatrix = Mat<f64>;

/// Sparse matrix wrapper (CSC) used for precision and operators.
#[derive(Clone, Debug)]
pub struct SparseMatrix {
    inner: SparseColMat<usize, f64>,
}

/// Sparse triplet used for COO-style assembly.
pub type SparseTriplet = Triplet<usize, usize, f64>;

/// Index in the original, user-facing coordinate order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OriginalIndex(pub usize);

/// Index in the fill-reducing Cholesky/factor coordinate order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PermutedIndex(pub usize);

/// Explicit permutation maps for sparse Cholesky factorizations.
///
/// The convention is `x_tilde = P x`, so `orig_to_perm[i]` gives the
/// permuted index for original coordinate `i`, while `perm_to_orig[k]` gives
/// the original coordinate for permuted index `k`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Permutation {
    pub orig_to_perm: Vec<usize>,
    pub perm_to_orig: Vec<usize>,
}

impl Permutation {
    pub fn new(orig_to_perm: Vec<usize>, perm_to_orig: Vec<usize>) -> Result<Self, GmrfError> {
        let permutation = Self {
            orig_to_perm,
            perm_to_orig,
        };
        permutation.validate()?;
        Ok(permutation)
    }

    pub fn identity(dimension: usize) -> Self {
        let indices = (0..dimension).collect::<Vec<_>>();
        Self {
            orig_to_perm: indices.clone(),
            perm_to_orig: indices,
        }
    }

    pub fn from_orig_to_perm(orig_to_perm: Vec<usize>) -> Result<Self, GmrfError> {
        let dimension = orig_to_perm.len();
        let mut seen = vec![false; dimension];
        let mut perm_to_orig = vec![0usize; dimension];
        for (original, permuted) in orig_to_perm.iter().copied().enumerate() {
            if permuted >= dimension {
                return Err(GmrfError::DimensionMismatch(
                    "permutation index exceeds dimension",
                ));
            }
            if seen[permuted] {
                return Err(GmrfError::DimensionMismatch(
                    "permutation contains duplicate entries",
                ));
            }
            seen[permuted] = true;
            perm_to_orig[permuted] = original;
        }
        Ok(Self {
            orig_to_perm,
            perm_to_orig,
        })
    }

    pub fn dimension(&self) -> usize {
        self.orig_to_perm.len()
    }

    pub fn validate(&self) -> Result<(), GmrfError> {
        let dimension = self.orig_to_perm.len();
        if self.perm_to_orig.len() != dimension {
            return Err(GmrfError::DimensionMismatch(
                "permutation arrays must have the same length",
            ));
        }
        let mut seen = vec![false; dimension];
        for (original, permuted) in self.orig_to_perm.iter().copied().enumerate() {
            if permuted >= dimension {
                return Err(GmrfError::DimensionMismatch(
                    "permutation index exceeds dimension",
                ));
            }
            if seen[permuted] {
                return Err(GmrfError::DimensionMismatch(
                    "permutation contains duplicate entries",
                ));
            }
            seen[permuted] = true;
            if self.perm_to_orig[permuted] != original {
                return Err(GmrfError::DimensionMismatch(
                    "permutation arrays are not inverses",
                ));
            }
        }
        Ok(())
    }

    pub fn permuted_index(&self, original: OriginalIndex) -> Result<PermutedIndex, GmrfError> {
        self.orig_to_perm
            .get(original.0)
            .copied()
            .map(PermutedIndex)
            .ok_or(GmrfError::DimensionMismatch(
                "original index exceeds permutation dimension",
            ))
    }

    pub fn original_index(&self, permuted: PermutedIndex) -> Result<OriginalIndex, GmrfError> {
        self.perm_to_orig
            .get(permuted.0)
            .copied()
            .map(OriginalIndex)
            .ok_or(GmrfError::DimensionMismatch(
                "permuted index exceeds permutation dimension",
            ))
    }
}

/// Simple COO builder for sparse matrices.
#[derive(Clone, Debug)]
pub struct CooMatrix {
    nrows: usize,
    ncols: usize,
    triplets: Vec<SparseTriplet>,
}

impl CooMatrix {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            triplets: Vec::new(),
        }
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn push(&mut self, row: usize, col: usize, value: f64) {
        self.triplets.push(SparseTriplet::new(row, col, value));
    }

    pub fn triplet_iter(&self) -> impl Iterator<Item = (usize, usize, &f64)> {
        self.triplets.iter().map(|t| (t.row, t.col, &t.val))
    }
}

impl From<&CooMatrix> for SparseMatrix {
    fn from(value: &CooMatrix) -> Self {
        let mat = SparseColMat::try_new_from_triplets(value.nrows, value.ncols, &value.triplets)
            .expect("invalid COO triplets");
        Self { inner: mat }
    }
}

impl From<SparseColMat<usize, f64>> for SparseMatrix {
    fn from(inner: SparseColMat<usize, f64>) -> Self {
        Self { inner }
    }
}

impl From<&SparseMatrix> for CooMatrix {
    fn from(value: &SparseMatrix) -> Self {
        let mut coo = CooMatrix::new(value.nrows(), value.ncols());
        for (row, col, val) in value.triplet_iter() {
            coo.push(row, col, *val);
        }
        coo
    }
}

impl SparseMatrix {
    pub fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    pub fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    pub fn nnz(&self) -> usize {
        self.inner.compute_nnz()
    }

    pub fn as_ref(&self) -> SparseColMatRef<'_, usize, f64> {
        self.inner.as_ref()
    }

    pub fn triplet_iter(&self) -> SparseTripletIter<'_> {
        SparseTripletIter::new(&self.inner)
    }

    pub fn mul_vec(&self, x: &Vector) -> Vector {
        let mut out = Vector::zeros(self.nrows());
        for (row, col, value) in self.triplet_iter() {
            out[row] += *value * x[col];
        }
        out
    }

    /// Apply a symmetric permutation `P A P^T`, returning the matrix in
    /// permuted coordinates.
    pub fn permute_symmetric(&self, permutation: &Permutation) -> Result<Self, GmrfError> {
        if self.nrows() != self.ncols() {
            return Err(GmrfError::DimensionMismatch(
                "symmetric permutation requires a square matrix",
            ));
        }
        if permutation.dimension() != self.nrows() {
            return Err(GmrfError::DimensionMismatch(
                "permutation dimension must match matrix dimension",
            ));
        }

        let mut coo = CooMatrix::new(self.nrows(), self.ncols());
        for (row, col, value) in self.triplet_iter() {
            coo.push(
                permutation.orig_to_perm[row],
                permutation.orig_to_perm[col],
                *value,
            );
        }
        Ok(Self::from(&coo))
    }

    /// Compute a sparse Cholesky factorization with a fill-reducing ordering.
    ///
    /// The returned factor stores the permutation internally, so it can be reused for
    /// solves and sampling without losing the ordering information.
    pub fn cholesky_sqrt_lower(&self) -> Result<SparseCholeskyFactor, GmrfError> {
        SparseCholeskyFactor::factorize(self)
    }

    /// Compute a sparse Cholesky factorization using the requested symmetric ordering.
    pub fn cholesky_sqrt_lower_with_ordering(
        &self,
        ordering: CholeskyOrdering,
    ) -> Result<SparseCholeskyFactor, GmrfError> {
        SparseCholeskyFactor::factorize_with_ordering(self, ordering)
    }

    /// Analyze the sparse Cholesky pattern using the requested symmetric ordering.
    ///
    /// The returned symbolic factorization can be reused for matrices with the
    /// exact same stored sparsity pattern and ordering.
    pub fn analyze_cholesky_with_ordering(
        &self,
        ordering: CholeskyOrdering,
    ) -> Result<SparseCholeskySymbolic, GmrfError> {
        SparseCholeskySymbolic::analyze_with_ordering(self, ordering)
    }

    /// Alias for `cholesky_sqrt_lower` with a clearer name.
    pub fn cholesky_factor(&self) -> Result<SparseCholeskyFactor, GmrfError> {
        self.cholesky_sqrt_lower()
    }

    /// Compute a sparse LU factorization with partial row pivoting.
    pub fn lu_factor(&self) -> Result<SparseLuFactor, GmrfError> {
        SparseLuFactor::factorize(self)
    }
}

/// Sparse Cholesky factorization with permutation support.
#[derive(Debug)]
pub struct SparseCholeskyFactor {
    symbolic: Arc<SymbolicCholesky<usize>>,
    values: Vec<f64>,
}

/// Reusable sparse Cholesky symbolic analysis for a fixed sparsity pattern.
#[derive(Clone, Debug)]
pub struct SparseCholeskySymbolic {
    symbolic: Arc<SymbolicCholesky<usize>>,
    side: Side,
    pattern: SparsePatternFingerprint,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct SparsePatternFingerprint {
    nrows: usize,
    ncols: usize,
    nnz: usize,
    hash: u64,
}

/// Fill-reducing ordering used for sparse Cholesky factorization.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum CholeskyOrdering {
    /// Approximate minimum degree ordering. This is faer's default.
    #[default]
    Amd,
    /// Keep the matrix in its input ordering.
    Identity,
    /// Use a caller-supplied permutation with the `x_tilde = P x` convention.
    Custom(Permutation),
}

/// Sparse LU factorization with partial row pivoting.
#[derive(Debug, Clone)]
pub struct SparseLuFactor {
    dimension: usize,
    factor: FaerSparseLu<usize, f64>,
}

impl SparseCholeskyFactor {
    /// Compute a fill-reducing sparse Cholesky factorization of `precision`.
    pub fn factorize(precision: &SparseMatrix) -> Result<Self, GmrfError> {
        Self::factorize_with_ordering(precision, CholeskyOrdering::default())
    }

    /// Compute a sparse Cholesky factorization of `precision` with an explicit ordering.
    pub fn factorize_with_ordering(
        precision: &SparseMatrix,
        ordering: CholeskyOrdering,
    ) -> Result<Self, GmrfError> {
        SparseCholeskySymbolic::analyze_with_ordering(precision, ordering)?.factor(precision)
    }

    fn factorize_with_symbolic(
        precision: &SparseMatrix,
        symbolic: Arc<SymbolicCholesky<usize>>,
        side: Side,
    ) -> Result<Self, GmrfError> {
        let mut values = vec![0.0; symbolic.len_val()];
        let par = get_global_parallelism();
        let mut mem =
            MemBuffer::new(symbolic.factorize_numeric_llt_scratch::<f64>(par, Default::default()));
        let stack = MemStack::new(&mut mem);
        symbolic
            .factorize_numeric_llt(
                &mut values,
                precision.as_ref(),
                side,
                Default::default(),
                par,
                stack,
                Default::default(),
            )
            .map_err(|_| GmrfError::NonPositiveDefinite)?;

        Ok(Self { symbolic, values })
    }

    pub fn dimension(&self) -> usize {
        self.symbolic.nrows()
    }
}

impl SparseCholeskySymbolic {
    /// Compute reusable symbolic Cholesky analysis with an explicit ordering.
    pub fn analyze_with_ordering(
        precision: &SparseMatrix,
        ordering: CholeskyOrdering,
    ) -> Result<Self, GmrfError> {
        if precision.nrows() != precision.ncols() {
            return Err(GmrfError::DimensionMismatch(
                "precision matrix must be square",
            ));
        }

        let side = select_cholesky_side(precision);
        let pattern = sparse_pattern_fingerprint(precision);
        let symbolic = match &ordering {
            CholeskyOrdering::Amd => factorize_symbolic_cholesky(
                precision.as_ref().symbolic(),
                side,
                SymmetricOrdering::Amd,
                CholeskySymbolicParams::default(),
            ),
            CholeskyOrdering::Identity => factorize_symbolic_cholesky(
                precision.as_ref().symbolic(),
                side,
                SymmetricOrdering::Identity,
                CholeskySymbolicParams::default(),
            ),
            CholeskyOrdering::Custom(permutation) => {
                if permutation.dimension() != precision.nrows() {
                    return Err(GmrfError::DimensionMismatch(
                        "custom ordering dimension must match precision dimension",
                    ));
                }
                permutation.validate()?;
                let custom = PermRef::new_checked(
                    &permutation.perm_to_orig,
                    &permutation.orig_to_perm,
                    permutation.dimension(),
                );
                factorize_symbolic_cholesky(
                    precision.as_ref().symbolic(),
                    side,
                    SymmetricOrdering::Custom(custom),
                    CholeskySymbolicParams::default(),
                )
            }
        }
        .map_err(|_| GmrfError::NonPositiveDefinite)?;

        Ok(Self {
            symbolic: Arc::new(symbolic),
            side,
            pattern,
        })
    }

    pub fn dimension(&self) -> usize {
        self.symbolic.nrows()
    }

    /// Numeric factorization using the previously analyzed symbolic pattern.
    pub fn factor(&self, precision: &SparseMatrix) -> Result<SparseCholeskyFactor, GmrfError> {
        if sparse_pattern_fingerprint(precision) != self.pattern {
            return Err(GmrfError::DimensionMismatch(
                "matrix sparsity pattern must match reusable symbolic Cholesky pattern",
            ));
        }
        SparseCholeskyFactor::factorize_with_symbolic(
            precision,
            Arc::clone(&self.symbolic),
            self.side,
        )
    }
}

impl SparseCholeskyFactor {
    /// Number of stored numeric entries in the Cholesky factor.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Return the factor permutation using the `x_tilde = P x` convention.
    pub fn permutation(&self) -> Permutation {
        match self.symbolic.perm() {
            Some(perm) => {
                let (perm_to_orig, orig_to_perm) = perm.arrays();
                Permutation {
                    orig_to_perm: orig_to_perm.to_vec(),
                    perm_to_orig: perm_to_orig.to_vec(),
                }
            }
            None => Permutation::identity(self.dimension()),
        }
    }

    /// Map a vector from original coordinates to Cholesky/permuted coordinates.
    pub fn permute_original_to_factor(&self, input: &Vector) -> Result<Vector, GmrfError> {
        if input.len() != self.dimension() {
            return Err(GmrfError::DimensionMismatch(
                "vector length must match factor dimension",
            ));
        }
        let permutation = self.permutation();
        Ok(Vector::from_fn(self.dimension(), |permuted| {
            input[permutation.perm_to_orig[permuted]]
        }))
    }

    /// Map a vector from Cholesky/permuted coordinates back to original coordinates.
    pub fn permute_factor_to_original(&self, input: &Vector) -> Result<Vector, GmrfError> {
        if input.len() != self.dimension() {
            return Err(GmrfError::DimensionMismatch(
                "vector length must match factor dimension",
            ));
        }
        let permutation = self.permutation();
        Ok(Vector::from_fn(self.dimension(), |original| {
            input[permutation.orig_to_perm[original]]
        }))
    }

    /// Solve `A x = rhs` in-place using the factorization.
    pub fn solve_in_place(&self, rhs: &mut Vector) -> Result<(), GmrfError> {
        if rhs.len() != self.dimension() {
            return Err(GmrfError::DimensionMismatch(
                "right hand side length must match precision dimension",
            ));
        }

        let par = get_global_parallelism();
        let mut mem = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(1, par));
        let stack = MemStack::new(&mut mem);
        let llt = LltRef::new(&self.symbolic, &self.values);
        llt.solve_in_place_with_conj(Conj::No, rhs.as_col_mut().as_mat_mut(), par, stack);
        Ok(())
    }

    /// Solve `A X = RHS` in-place for multiple right hand sides.
    pub fn solve_dense_in_place(&self, rhs: &mut DenseMatrix) -> Result<(), GmrfError> {
        if rhs.nrows() != self.dimension() {
            return Err(GmrfError::DimensionMismatch(
                "right hand side row count must match precision dimension",
            ));
        }

        let par = get_global_parallelism();
        let mut mem = MemBuffer::new(
            self.symbolic
                .solve_in_place_scratch::<f64>(rhs.ncols(), par),
        );
        let stack = MemStack::new(&mut mem);
        let llt = LltRef::new(&self.symbolic, &self.values);
        llt.solve_in_place_with_conj(Conj::No, rhs.as_mut(), par, stack);
        Ok(())
    }

    /// Solve `A x = rhs`, returning the solution.
    pub fn solve(&self, rhs: &Vector) -> Result<Vector, GmrfError> {
        let mut out = rhs.clone();
        self.solve_in_place(&mut out)?;
        Ok(out)
    }

    /// Solve `L x = rhs` in permuted factor coordinates.
    pub fn solve_l_permuted_in_place(&self, rhs: &mut Vector) -> Result<(), GmrfError> {
        if rhs.len() != self.dimension() {
            return Err(GmrfError::DimensionMismatch(
                "right hand side length must match precision dimension",
            ));
        }
        self.solve_l_permuted_in_place_inner(rhs)
    }

    /// Solve `Lᵀ x = rhs` in permuted factor coordinates.
    pub fn solve_l_transpose_permuted_in_place(&self, rhs: &mut Vector) -> Result<(), GmrfError> {
        if rhs.len() != self.dimension() {
            return Err(GmrfError::DimensionMismatch(
                "right hand side length must match precision dimension",
            ));
        }
        self.solve_l_transpose_in_place_inner(rhs)
    }

    /// Solve `Q_tilde x = rhs`, where `Q_tilde = P Q P^T`, in permuted coordinates.
    pub fn solve_permuted_in_place(&self, rhs: &mut Vector) -> Result<(), GmrfError> {
        self.solve_l_permuted_in_place(rhs)?;
        self.solve_l_transpose_permuted_in_place(rhs)
    }

    /// Solve `Q_tilde x = rhs`, returning the result in permuted coordinates.
    pub fn solve_permuted(&self, rhs: &Vector) -> Result<Vector, GmrfError> {
        let mut out = rhs.clone();
        self.solve_permuted_in_place(&mut out)?;
        Ok(out)
    }

    /// Multiply by the sparse Cholesky factor `L` in permuted coordinates.
    pub fn mul_l_permuted(&self, input: &Vector) -> Result<Vector, GmrfError> {
        if input.len() != self.dimension() {
            return Err(GmrfError::DimensionMismatch(
                "input length must match precision dimension",
            ));
        }
        let mut out = Vector::zeros(self.dimension());
        for (row, col, value) in self.lower_triplets() {
            out[row] += value * input[col];
        }
        Ok(out)
    }

    /// Return lower-triangular Cholesky entries `(row, col, value)` in permuted coordinates.
    pub fn lower_triplets(&self) -> Vec<(usize, usize, f64)> {
        match self.symbolic.raw() {
            SymbolicCholeskyRaw::Simplicial(symbolic) => {
                let factor = symbolic.factor();
                let col_ptr = factor.col_ptr();
                let row_idx = factor.row_idx();
                let mut entries = Vec::with_capacity(self.values.len());
                for col in 0..self.dimension() {
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    for (&row, &value) in row_idx[start..end]
                        .iter()
                        .zip(self.values[start..end].iter())
                    {
                        if row >= col && value != 0.0 {
                            entries.push((row, col, value));
                        }
                    }
                }
                entries
            }
            SymbolicCholeskyRaw::Supernodal(symbolic) => {
                let llt = SupernodalLltRef::new(symbolic, &self.values);
                let mut entries = Vec::with_capacity(self.values.len());
                for supernode_index in 0..symbolic.n_supernodes() {
                    let node = llt.supernode(supernode_index);
                    let matrix = node.val();
                    let symbolic_node = symbolic.supernode(supernode_index);
                    let start = symbolic_node.start();
                    let size = matrix.ncols();
                    let pattern = symbolic_node.pattern();
                    for local_col in 0..size {
                        let global_col = start + local_col;
                        for local_row in local_col..matrix.nrows() {
                            let global_row = if local_row < size {
                                start + local_row
                            } else {
                                pattern[local_row - size]
                            };
                            let row = unsafe { faer::Idx::<usize>::new_unbound(local_row) };
                            let col = unsafe { faer::Idx::<usize>::new_unbound(local_col) };
                            let value = matrix[(row, col)];
                            if global_row >= global_col && value != 0.0 {
                                entries.push((global_row, global_col, value));
                            }
                        }
                    }
                }
                entries
            }
        }
    }

    /// Solve `Lᵀ x = rhs` in-place using the factorization.
    pub fn solve_l_transpose_in_place(&self, rhs: &mut Vector) -> Result<(), GmrfError> {
        if rhs.len() != self.dimension() {
            return Err(GmrfError::DimensionMismatch(
                "right hand side length must match precision dimension",
            ));
        }

        match self.symbolic.perm() {
            Some(perm) => {
                let n = rhs.len();
                let mut permuted = Vector::zeros(n);
                for (i, fwd) in perm.arrays().0.iter().enumerate() {
                    permuted[i] = rhs[*fwd];
                }
                self.solve_l_transpose_in_place_inner(&mut permuted)?;
                for (i, inv) in perm.arrays().1.iter().enumerate() {
                    rhs[i] = permuted[*inv];
                }
            }
            None => {
                self.solve_l_transpose_in_place_inner(rhs)?;
            }
        }

        Ok(())
    }

    fn solve_l_transpose_in_place_inner(&self, rhs: &mut Vector) -> Result<(), GmrfError> {
        match self.symbolic.raw() {
            SymbolicCholeskyRaw::Simplicial(symbolic) => {
                let l = SparseColMatRef::new(symbolic.factor(), &self.values);
                l.transpose()
                    .sp_solve_upper_triangular_in_place(rhs.as_col_mut().as_mat_mut());
            }
            SymbolicCholeskyRaw::Supernodal(symbolic) => {
                let par = get_global_parallelism();
                let mut mem = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(1, par));
                let stack = MemStack::new(&mut mem);
                let llt = SupernodalLltRef::new(symbolic, &self.values);
                llt.l_transpose_solve_with_conj(
                    Conj::No,
                    rhs.as_col_mut().as_mat_mut(),
                    par,
                    stack,
                );
            }
        }

        Ok(())
    }

    fn solve_l_permuted_in_place_inner(&self, rhs: &mut Vector) -> Result<(), GmrfError> {
        match self.symbolic.raw() {
            SymbolicCholeskyRaw::Simplicial(symbolic) => {
                let l = SparseColMatRef::new(symbolic.factor(), &self.values);
                l.sp_solve_lower_triangular_in_place(rhs.as_col_mut().as_mat_mut());
            }
            SymbolicCholeskyRaw::Supernodal(symbolic) => {
                let par = get_global_parallelism();
                let mut mem = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(1, par));
                let stack = MemStack::new(&mut mem);
                let llt = SupernodalLltRef::new(symbolic, &self.values);
                llt.l_solve_with_conj(Conj::No, rhs.as_col_mut().as_mat_mut(), par, stack);
            }
        }

        Ok(())
    }

    /// Compute `log(det(Q))` from the Cholesky factor.
    pub fn logdet_precision(&self) -> Result<f64, GmrfError> {
        let mut acc = 0.0;
        match self.symbolic.raw() {
            SymbolicCholeskyRaw::Simplicial(symbolic) => {
                let factor = symbolic.factor();
                let col_ptr = factor.col_ptr();
                for col in 0..self.dimension() {
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    if start == end {
                        return Err(GmrfError::NonPositiveDefinite);
                    }
                    let diag = self.values[start];
                    if diag <= 0.0 {
                        return Err(GmrfError::NonPositiveDefinite);
                    }
                    acc += diag.ln();
                }
            }
            SymbolicCholeskyRaw::Supernodal(symbolic) => {
                let llt = SupernodalLltRef::new(symbolic, &self.values);
                for s in 0..symbolic.n_supernodes() {
                    let node = llt.supernode(s);
                    let matrix = node.val();
                    let size = matrix.ncols();
                    let (top, _) = matrix.split_at_row(size);
                    for i in 0..size {
                        let idx = unsafe { faer::Idx::<usize>::new_unbound(i) };
                        let diag = top[(idx, idx)];
                        if diag <= 0.0 {
                            return Err(GmrfError::NonPositiveDefinite);
                        }
                        acc += diag.ln();
                    }
                }
            }
        }
        Ok(2.0 * acc)
    }
}

impl SparseLuFactor {
    /// Compute a sparse LU factorization of `matrix`.
    pub fn factorize(matrix: &SparseMatrix) -> Result<Self, GmrfError> {
        if matrix.nrows() != matrix.ncols() {
            return Err(GmrfError::DimensionMismatch("matrix must be square"));
        }

        let factor = matrix
            .as_ref()
            .sp_lu()
            .map_err(|_| GmrfError::SingularMatrix)?;
        Ok(Self {
            dimension: matrix.nrows(),
            factor,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Solve `A x = rhs` in-place using the factorization.
    pub fn solve_in_place(&self, rhs: &mut Vector) -> Result<(), GmrfError> {
        if rhs.len() != self.dimension {
            return Err(GmrfError::DimensionMismatch(
                "right hand side length must match matrix dimension",
            ));
        }

        self.factor.solve_in_place(rhs.as_col_mut().as_mat_mut());
        Ok(())
    }

    /// Solve `A x = rhs`, returning the solution.
    pub fn solve(&self, rhs: &Vector) -> Result<Vector, GmrfError> {
        let mut out = rhs.clone();
        self.solve_in_place(&mut out)?;
        Ok(out)
    }
}

fn select_cholesky_side(precision: &SparseMatrix) -> Side {
    let mut has_upper = false;
    for (row, col, _) in precision.triplet_iter() {
        if row < col {
            has_upper = true;
        }
        if has_upper {
            break;
        }
    }
    if has_upper {
        Side::Upper
    } else {
        Side::Lower
    }
}

fn sparse_pattern_fingerprint(matrix: &SparseMatrix) -> SparsePatternFingerprint {
    let mut hasher = DefaultHasher::new();
    matrix.nrows().hash(&mut hasher);
    matrix.ncols().hash(&mut hasher);
    let mut nnz = 0usize;
    for (row, col, _) in matrix.triplet_iter() {
        row.hash(&mut hasher);
        col.hash(&mut hasher);
        nnz += 1;
    }
    SparsePatternFingerprint {
        nrows: matrix.nrows(),
        ncols: matrix.ncols(),
        nnz,
        hash: hasher.finish(),
    }
}

/// Iterator over triplets in a CSC matrix.
pub struct SparseTripletIter<'a> {
    mat: &'a SparseColMat<usize, f64>,
    col: usize,
    idx: usize,
}

impl<'a> SparseTripletIter<'a> {
    fn new(mat: &'a SparseColMat<usize, f64>) -> Self {
        Self {
            mat,
            col: 0,
            idx: 0,
        }
    }
}

impl<'a> Iterator for SparseTripletIter<'a> {
    type Item = (usize, usize, &'a f64);

    fn next(&mut self) -> Option<Self::Item> {
        let ncols = self.mat.ncols();
        let col_ptr = self.mat.col_ptr();
        let row_idx = self.mat.row_idx();
        let vals = self.mat.val();

        while self.col < ncols {
            let end = col_ptr[self.col + 1];
            if self.idx < end {
                let row = row_idx[self.idx];
                let val = &vals[self.idx];
                let col = self.col;
                self.idx += 1;
                return Some((row, col, val));
            }
            self.col += 1;
            if self.col < ncols {
                self.idx = col_ptr[self.col];
            }
        }
        None
    }
}

/// Dense vector used throughout the crate.
#[derive(Clone, Debug, PartialEq)]
pub struct Vector {
    data: Vec<f64>,
}

impl Vector {
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![0.0; len],
        }
    }

    pub fn from_element(len: usize, value: f64) -> Self {
        Self {
            data: vec![value; len],
        }
    }

    pub fn from_vec(data: Vec<f64>) -> Self {
        Self { data }
    }

    pub fn from_fn(len: usize, mut f: impl FnMut(usize) -> f64) -> Self {
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            data.push(f(i));
        }
        Self { data }
    }

    pub fn from_iterator<I: IntoIterator<Item = f64>>(len: usize, iter: I) -> Self {
        let mut data = Vec::with_capacity(len);
        data.extend(iter);
        Self { data }
    }

    pub fn from_column_slice(slice: &[f64]) -> Self {
        Self {
            data: slice.to_vec(),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.data.iter()
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    pub fn dot(&self, other: &Self) -> f64 {
        assert_eq!(self.len(), other.len());
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn component_mul(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Self { data }
    }

    pub fn component_div(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a / b)
            .collect();
        Self { data }
    }

    pub fn map(&self, f: impl FnMut(f64) -> f64) -> Self {
        let data = self.data.iter().copied().map(f).collect();
        Self { data }
    }

    pub fn as_col(&self) -> faer::ColRef<'_, f64> {
        faer::ColRef::from_slice(&self.data)
    }

    pub fn as_col_mut(&mut self) -> faer::ColMut<'_, f64> {
        faer::ColMut::from_slice_mut(&mut self.data)
    }
}

impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<'b> Add<&'b Vector> for &Vector {
    type Output = Vector;

    fn add(self, rhs: &'b Vector) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Vector { data }
    }
}

impl Add<Vector> for &Vector {
    type Output = Vector;

    fn add(self, rhs: Vector) -> Self::Output {
        self + &rhs
    }
}

impl<'a> Add<&'a Vector> for Vector {
    type Output = Vector;

    fn add(self, rhs: &'a Vector) -> Self::Output {
        &self + rhs
    }
}

impl Add for Vector {
    type Output = Vector;

    fn add(self, rhs: Vector) -> Self::Output {
        &self + &rhs
    }
}

impl<'b> Sub<&'b Vector> for &Vector {
    type Output = Vector;

    fn sub(self, rhs: &'b Vector) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Vector { data }
    }
}

impl Sub<Vector> for &Vector {
    type Output = Vector;

    fn sub(self, rhs: Vector) -> Self::Output {
        self - &rhs
    }
}

impl<'a> Sub<&'a Vector> for Vector {
    type Output = Vector;

    fn sub(self, rhs: &'a Vector) -> Self::Output {
        &self - rhs
    }
}

impl Sub for Vector {
    type Output = Vector;

    fn sub(self, rhs: Vector) -> Self::Output {
        &self - &rhs
    }
}

impl AddAssign<&Vector> for Vector {
    fn add_assign(&mut self, rhs: &Vector) {
        assert_eq!(self.len(), rhs.len());
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a += b;
        }
    }
}

impl AddAssign<Vector> for Vector {
    fn add_assign(&mut self, rhs: Vector) {
        *self += &rhs;
    }
}

impl SubAssign<&Vector> for Vector {
    fn sub_assign(&mut self, rhs: &Vector) {
        assert_eq!(self.len(), rhs.len());
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a -= b;
        }
    }
}

impl SubAssign<Vector> for Vector {
    fn sub_assign(&mut self, rhs: Vector) {
        *self -= &rhs;
    }
}

impl Mul<f64> for &Vector {
    type Output = Vector;

    fn mul(self, rhs: f64) -> Self::Output {
        let data = self.data.iter().map(|v| v * rhs).collect();
        Vector { data }
    }
}

impl Mul<f64> for Vector {
    type Output = Vector;

    fn mul(self, rhs: f64) -> Self::Output {
        &self * rhs
    }
}

impl Mul<&Vector> for f64 {
    type Output = Vector;

    fn mul(self, rhs: &Vector) -> Self::Output {
        rhs * self
    }
}

impl Mul<Vector> for f64 {
    type Output = Vector;

    fn mul(self, rhs: Vector) -> Self::Output {
        &rhs * self
    }
}

impl<'a> Mul<&'a Vector> for &'a SparseMatrix {
    type Output = Vector;

    fn mul(self, rhs: &'a Vector) -> Self::Output {
        self.mul_vec(rhs)
    }
}

impl Div<f64> for &Vector {
    type Output = Vector;

    fn div(self, rhs: f64) -> Self::Output {
        let data = self.data.iter().map(|v| v / rhs).collect();
        Vector { data }
    }
}

impl Div<f64> for Vector {
    type Output = Vector;

    fn div(self, rhs: f64) -> Self::Output {
        &self / rhs
    }
}

impl FromIterator<f64> for Vector {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        Self {
            data: iter.into_iter().collect(),
        }
    }
}

/// Error variants produced by the core GMRF routines.
#[derive(Debug, Error)]
pub enum GmrfError {
    /// The provided inputs are inconsistent (dimension mismatch or missing data).
    #[error("inconsistent dimensions: {0}")]
    DimensionMismatch(&'static str),

    /// A precision matrix was required but not available in concrete form.
    #[error("precision matrix is required for this operation")]
    MissingPrecisionMatrix,

    /// Exact marginal variance recovery requires an explicit sparse precision matrix.
    #[error("exact variance diagonal requires an explicit sparse precision matrix")]
    ExactVarianceRequiresPrecisionMatrix,

    /// A precision factorization was required but not available.
    #[error("precision factorization is required for this operation")]
    MissingPrecisionSqrt,

    /// A requested selected-inverse entry is not covered by the computed sparse inverse pattern.
    #[error("selected inverse entry is not covered by the computed Takahashi closure")]
    SelectedInversePatternNotCovered,

    /// A sparse matrix factorization failed because the matrix was singular.
    #[error("matrix is singular")]
    SingularMatrix,

    /// Factorization failed because the precision was not positive definite.
    #[error("precision matrix is not positive definite")]
    NonPositiveDefinite,

    /// Linear equality constraints were singular under the prior covariance.
    #[error("linear constraints are singular or not full row-rank")]
    SingularConstraintSystem,

    /// A covariance-derived quantity violated basic positivity constraints beyond roundoff.
    #[error("numerical instability while computing constrained covariance: {0}")]
    NumericalInstability(&'static str),

    /// An iterative linear solver reached its iteration limit before convergence.
    #[error(
        "iterative solver did not converge after {iterations} iterations; final residual norm {residual_norm:.6e}"
    )]
    IterativeSolverDidNotConverge {
        iterations: usize,
        residual_norm: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::{CholeskyOrdering, CooMatrix, Permutation, SparseMatrix, Vector};

    #[test]
    fn permutation_validation_accepts_bijections() {
        let identity = Permutation::from_orig_to_perm(vec![0, 1, 2]).unwrap();
        assert_eq!(identity.perm_to_orig, vec![0, 1, 2]);

        let permutation = Permutation::from_orig_to_perm(vec![2, 0, 1]).unwrap();
        assert_eq!(permutation.perm_to_orig, vec![1, 2, 0]);
        assert!(Permutation::new(vec![2, 0, 1], vec![1, 2, 0]).is_ok());
    }

    #[test]
    fn permutation_validation_rejects_invalid_arrays() {
        assert!(Permutation::from_orig_to_perm(vec![0, 0, 2]).is_err());
        assert!(Permutation::from_orig_to_perm(vec![0, 1, 3]).is_err());
        assert!(Permutation::new(vec![0, 1, 2], vec![0, 2, 1]).is_err());
        assert!(Permutation::new(vec![0, 1, 2], vec![0, 1]).is_err());
    }

    #[test]
    fn cholesky_factor_solves_linear_system() {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 4.0);
        coo.push(0, 1, 1.0);
        coo.push(1, 0, 1.0);
        coo.push(1, 1, 3.0);
        coo.push(1, 2, 1.0);
        coo.push(2, 1, 1.0);
        coo.push(2, 2, 2.0);
        let q = SparseMatrix::from(&coo);

        let factor = q.cholesky_sqrt_lower().unwrap();
        let x = Vector::from_vec(vec![0.5, -1.2, 0.7]);
        let b = q.mul_vec(&x);
        let mut solved = b.clone();
        factor.solve_in_place(&mut solved).unwrap();
        let diff = (solved - x).norm();
        assert!(diff < 1e-10);
    }

    #[test]
    fn cholesky_factor_reports_numeric_nnz() {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 4.0);
        coo.push(0, 1, 1.0);
        coo.push(1, 0, 1.0);
        coo.push(1, 1, 3.0);
        coo.push(1, 2, 1.0);
        coo.push(2, 1, 1.0);
        coo.push(2, 2, 2.0);
        let q = SparseMatrix::from(&coo);

        let factor = q.cholesky_sqrt_lower().unwrap();
        assert_eq!(factor.dimension(), 3);
        assert_eq!(factor.nnz(), 5);
    }

    #[test]
    fn cholesky_factor_accepts_explicit_ordering() {
        let mut coo = CooMatrix::new(4, 4);
        for index in 0..4 {
            coo.push(index, index, 3.0);
            if index + 1 < 4 {
                coo.push(index, index + 1, -1.0);
                coo.push(index + 1, index, -1.0);
            }
        }
        let q = SparseMatrix::from(&coo);

        let amd = q
            .cholesky_sqrt_lower_with_ordering(CholeskyOrdering::Amd)
            .unwrap();
        let identity = q
            .cholesky_sqrt_lower_with_ordering(CholeskyOrdering::Identity)
            .unwrap();

        assert_eq!(amd.dimension(), 4);
        assert_eq!(identity.dimension(), 4);
    }

    #[test]
    fn cholesky_factor_accepts_custom_ordering() {
        let mut coo = CooMatrix::new(4, 4);
        for index in 0..4 {
            coo.push(index, index, 3.0);
            if index + 1 < 4 {
                coo.push(index, index + 1, -1.0);
                coo.push(index + 1, index, -1.0);
            }
        }
        let q = SparseMatrix::from(&coo);
        let ordering =
            CholeskyOrdering::Custom(Permutation::from_orig_to_perm(vec![2, 0, 3, 1]).unwrap());
        let factor = q.cholesky_sqrt_lower_with_ordering(ordering).unwrap();

        let x = Vector::from_vec(vec![0.25, -0.5, 1.25, 0.75]);
        let b = q.mul_vec(&x);
        let mut solved = b.clone();
        factor.solve_in_place(&mut solved).unwrap();
        assert!((solved - x).norm() < 1e-10);
    }

    #[test]
    fn reusable_cholesky_symbolic_matches_direct_factorization() {
        let mut coo = CooMatrix::new(4, 4);
        for index in 0..4 {
            coo.push(index, index, 4.0);
            if index + 1 < 4 {
                coo.push(index, index + 1, -1.0);
                coo.push(index + 1, index, -1.0);
            }
        }
        let q = SparseMatrix::from(&coo);
        let ordering =
            CholeskyOrdering::Custom(Permutation::from_orig_to_perm(vec![2, 0, 3, 1]).unwrap());
        let symbolic = q.analyze_cholesky_with_ordering(ordering.clone()).unwrap();
        let reused = symbolic.factor(&q).unwrap();
        let direct = q.cholesky_sqrt_lower_with_ordering(ordering).unwrap();

        let x = Vector::from_vec(vec![0.25, -0.5, 1.25, 0.75]);
        let b = q.mul_vec(&x);
        assert!((reused.solve(&b).unwrap() - &x).norm() < 1e-10);
        assert_eq!(reused.nnz(), direct.nnz());
    }

    #[test]
    fn reusable_cholesky_symbolic_rejects_pattern_mismatch() {
        let mut base = CooMatrix::new(3, 3);
        base.push(0, 0, 3.0);
        base.push(1, 1, 3.0);
        base.push(2, 2, 3.0);
        let q = SparseMatrix::from(&base);
        let symbolic = q
            .analyze_cholesky_with_ordering(CholeskyOrdering::Identity)
            .unwrap();

        let mut changed = CooMatrix::new(3, 3);
        changed.push(0, 0, 3.0);
        changed.push(0, 1, -1.0);
        changed.push(1, 0, -1.0);
        changed.push(1, 1, 3.0);
        changed.push(2, 2, 3.0);
        let changed = SparseMatrix::from(&changed);

        assert!(symbolic.factor(&changed).is_err());
    }

    #[test]
    fn lu_factor_solves_nonsymmetric_linear_system() {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 0.0);
        coo.push(0, 1, 2.0);
        coo.push(0, 2, 1.0);
        coo.push(1, 0, 1.0);
        coo.push(1, 1, 1.0);
        coo.push(2, 0, 2.0);
        coo.push(2, 2, 1.0);
        let a = SparseMatrix::from(&coo);
        let factor = a.lu_factor().unwrap();

        let x = Vector::from_vec(vec![1.0, -2.0, 0.5]);
        let b = a.mul_vec(&x);
        let solved = factor.solve(&b).unwrap();
        assert!((solved - x).norm() < 1e-10);
    }

    #[test]
    fn lu_factor_reuses_factorization_across_multiple_rhs() {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 4.0);
        coo.push(0, 1, -1.0);
        coo.push(1, 0, 2.0);
        coo.push(1, 1, 3.0);
        coo.push(1, 2, 1.0);
        coo.push(2, 0, 1.0);
        coo.push(2, 2, 2.0);
        let a = SparseMatrix::from(&coo);
        let factor = a.lu_factor().unwrap();

        let x1 = Vector::from_vec(vec![0.25, -1.0, 2.0]);
        let x2 = Vector::from_vec(vec![-0.5, 1.5, 0.75]);
        let b1 = a.mul_vec(&x1);
        let b2 = a.mul_vec(&x2);

        let solved1 = factor.solve(&b1).unwrap();
        let solved2 = factor.solve(&b2).unwrap();
        assert!((solved1 - x1).norm() < 1e-10);
        assert!((solved2 - x2).norm() < 1e-10);
    }
}
