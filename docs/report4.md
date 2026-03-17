# Block Algorithm for Dense vs Banded Matrices

## The LAPACK Milestone

In the late 1980s, LAPACK introduced **block algorithms** — one of the most
important performance breakthroughs in numerical linear algebra history.

The insight: reorganize matrix factorization from column-by-column elimination
(BLAS Level 2, matrix-vector) into panel factorization + trailing matrix update
using matrix-matrix multiply (BLAS Level 3). BLAS-3 operations reuse each data
element O(NB) times in cache, converting memory-bound code into compute-bound code.

```
Point algorithm (BLAS-2):        Blocked algorithm (BLAS-3):
  for each column k:               for each block of NB columns:
    eliminate below diagonal          1. Panel: factor NB columns (BLAS-2)
    update ALL remaining columns      2. DTRSM: solve L11 * U12 = A12
                                      3. DGEMM: A22 -= L21 * U12  ← cache win
```

## Implementation

We implemented the blocked dense LU decomposition with three phases per block:

1. **Panel factorization** — point LU on NB columns with partial pivoting,
   updating only within-panel columns
2. **DTRSM** — forward-substitute L11 into trailing columns for panel rows,
   producing the correct U12
3. **DGEMM** — update trailing submatrix: A22 -= L21 * U12, using
   row-major-optimal loop order `(i, j, c)` where the innermost loop
   sweeps a row of U12 sequentially

The `(i, j, c)` loop order is critical for row-major storage: load L21[i][j]
once as a scalar, then sweep the entire U12 row j — every access is to the
next cache line. The naive `(i, c, j)` order accesses U12 column-by-column,
jumping N doubles per access (cache-hostile).

## Dense Matrix Results

Environment: Windows 11, MSVC 19.44 (VS2022), /O2, single-threaded.

### Point vs Blocked (NB=64)

| N     | Point (ms) | Blocked (ms) | Speedup  | Residual |
|-------|-----------|--------------|----------|----------|
| 100   | 0.14      | 0.10         | **1.4x** | identical |
| 200   | 0.91      | 0.60         | **1.5x** | identical |
| 500   | 14.9      | 9.17         | **1.6x** | identical |
| 1,000 | 126       | 71           | **1.8x** | identical |
| 2,000 | 2,124     | 601          | **3.5x** | identical |

**Speedup grows with N** — exactly as theory predicts:
- At N=100: matrix is 78 KB, fits in L2 cache → small benefit
- At N=1000: matrix is 7.6 MB, exceeds L2 → 1.8x
- At N=2000: matrix is 30.5 MB, far exceeds all cache → **3.5x**

### Optimal Block Size

| NB (N=1000) | Total (ms) | NB (N=2000) | Total (ms) |
|-------------|-----------|-------------|-----------|
| point       | 133       | point       | 2,007     |
| 8           | 81        | 16          | 637       |
| 16          | 78        | 32          | 596       |
| **32**      | **76**    | **64**      | **584**   |
| 64          | 81        | 128         | 676       |
| 128         | 76        | 256         | 685       |

Best NB = **32–64**. Too small: DGEMM doesn't fill cache lines. Too large:
panel factorization dominates (still BLAS-2).

## Why Blocking Helps Dense But Not Banded

We also implemented the blocked algorithm for banded matrices (report3.md).
The contrast is striking:

| Matrix Type | Working Set (N=2000)  | Blocked vs Point |
|-------------|-----------------------|------------------|
| Dense       | 30.5 MB (N^2 doubles) | **3.5x faster**  |
| Banded (b=3)| 112 KB (N*7 doubles)  | 3.8x **slower**  |
| Banded (b=64)| 2 MB (N*129 doubles) | 8.2x **slower**  |

The reason is **cache residency**:

```
Dense matrix:  N^2 doubles → far exceeds cache
  → DGEMM reuses data NB times in cache → huge win

Banded matrix: N * bandwidth doubles → fits in L1/L2 cache
  → Point algorithm already has full cache reuse
  → Blocking adds overhead (DTRSM, index math) with zero cache benefit
```

### The Cache Threshold

| Working set size | Cache level | Blocking benefit |
|------------------|-------------|------------------|
| < 32 KB          | Fits in L1  | None (overhead hurts) |
| 32 KB – 256 KB   | Fits in L2  | Marginal (1.1-1.4x) |
| 256 KB – 8 MB    | Fits in L3  | Moderate (1.5-2x) |
| > 8 MB           | Main memory | **Large (2-4x+)** |

Banded matrices with bandwidth < 4000 always fit in L2 cache.
Dense matrices exceed L2 at N > ~180.

## Performance Hierarchy — Complete Picture

Combining all optimizations explored in this project:

### For Banded Systems (band=3)

| Optimization                         | Impact vs baseline |
|--------------------------------------|--------------------|
| 1. Banded algorithm (vs dense)       | **4,800x**         |
| 2. vs Eigen Sparse LU                | **23x**            |
| 3. Branch removal + reciprocal precomp| **24%**           |
| 4. SIMD / AVX-512                    | **~8%**            |
| 5. Block algorithm                   | **no benefit**     |

### For Dense Systems

| Optimization                         | Impact vs baseline |
|--------------------------------------|--------------------|
| 1. Block algorithm (NB=64, N=2000)   | **3.5x**           |
| 2. Row-major loop order in DGEMM     | **3x** (vs naive column order) |
| 3. SIMD / AVX (via compiler)         | included in above  |

## Files

| File                              | Description                          |
|-----------------------------------|--------------------------------------|
| `src/dense_matrix.h`             | Added `dense_lu_decompose_blocked()` |
| `src/dense_matrix.cpp`           | Blocked dense LU implementation      |
| `test/bench_dense_blocked.cpp`   | Dense point vs blocked benchmark     |
| `src/banded_matrix.h`            | `BandMatrixBlocked` struct           |
| `src/banded_matrix.cpp`          | Blocked banded LU (for comparison)   |
| `test/bench_blocked.cpp`         | Banded point vs blocked benchmark    |

## Conclusion

The LAPACK block algorithm is a **proven win for dense matrices** — 3.5x at
N=2000, growing further with N. The key is the DGEMM trailing update that
reuses panel data NB times in cache, overcoming the memory bandwidth wall.

For **banded matrices**, blocking provides no benefit because the compact
band storage already ensures cache residency. The point algorithm (NR3's
`bandec`) with micro-optimizations (branch removal, reciprocal precomputation)
is optimal.

**The lesson**: the right optimization depends on the data structure.
Dense matrices need algorithmic restructuring (blocking) to overcome cache
limits. Banded matrices need compact storage to avoid the cache problem
entirely. Both approaches achieve the same goal — keeping data in cache —
through different means.
