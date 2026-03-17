# Block Algorithm Analysis for Banded Matrix Solvers

## LAPACK Block Algorithm — History

One of the most important milestones in the history of BLAS/LAPACK was the
transition from **point algorithms** (BLAS Level 2) to **block algorithms** (BLAS
Level 3) in the late 1980s.

### The Problem

Point algorithms process one column at a time. For dense LU decomposition,
each column elimination is a matrix-vector operation (BLAS-2): O(N^2) data
for O(N^2) operations, giving an arithmetic intensity of O(1). This means
the algorithm is **memory-bandwidth bound** — the CPU spends most of its
time waiting for data from memory.

### The Solution

Block algorithms reorganize the factorization into two phases per block:

1. **Panel factorization** (BLAS-2): Factor NB columns using the point
   algorithm. This produces a tall-skinny L panel and a wide U panel.

2. **Trailing matrix update** (BLAS-3): Apply the L panel to the remaining
   matrix using DGEMM (matrix-matrix multiply). DGEMM has O(NB) arithmetic
   intensity — it reuses each data element NB times, keeping data in cache.

For dense matrices, this was transformative. On the IBM RISC System/6000 (1990),
the point algorithm achieved only a fraction of peak FLOPS, while the blocked
algorithm ran near peak due to DGEMM's cache-optimal access pattern.

## Implementation

We implemented a blocked banded LU decomposition following the LAPACK `dgbtrf`
approach:

- **LAPACK-style storage**: `ldab = 2*m1 + m2 + 1`, no row-shift. Element
  A(i,j) at `ab[i * ldab + (m1 + j - i)]`. Extra m1 columns for fill-in
  from pivoting.

- **Panel factorization**: Point algorithm on NB columns, updating only
  within-panel columns.

- **DTRSM**: Forward-substitute L11 into trailing columns for panel rows,
  computing U12 = L11^{-1} * A12.

- **DGEMM**: Update below-panel rows: A22 -= L21 * U12.

## Benchmark Results

Environment: Windows 11, MSVC 19.44 (VS2022), /O2, N = 100,000.

### Point vs Blocked (NB=32) Across Bandwidths

| Band | Point (ms) | Blocked (ms) | Blocked/Point | Residual Match |
|------|-----------|--------------|---------------|----------------|
| 1    | 1.92      | 3.38         | 1.8x slower   | identical       |
| 3    | 2.98      | 11.20        | 3.8x slower   | identical       |
| 8    | 7.41      | 46.06        | 6.2x slower   | identical       |
| 16   | 18.08     | 152.18       | 8.4x slower   | identical       |
| 32   | 62.66     | 615.47       | 9.8x slower   | identical       |
| 64   | 232.83    | 1901.84      | 8.2x slower   | identical       |

The blocked algorithm produces **identical residuals** (correctness verified)
but is **consistently 2-10x slower** than the point algorithm.

### Varying Block Size (band=32)

| NB    | Total (ms) | vs Point (63 ms) |
|-------|-----------|-------------------|
| point | 63.33     | 1.0x              |
| 4     | 512.52    | 8.1x slower       |
| 8     | 449.98    | 7.1x slower       |
| 16    | 474.35    | 7.5x slower       |
| 32    | 599.33    | 9.5x slower       |
| 64    | 730.97    | 11.5x slower      |

No block size helps. The overhead is fundamental, not a tuning issue.

## Why Blocking Doesn't Help Banded Matrices

The LAPACK block algorithm was a landmark for **dense** matrices. It does
**not** help banded matrices. Here is why:

### 1. The Working Set Already Fits in Cache

For a banded matrix with bandwidth `bw = 2*band + 1`:

| Band | bw  | Working set per step | L1 cache |
|------|-----|---------------------|----------|
| 3    | 7   | 7 * 8B = 56 bytes   | 32-64 KB |
| 16   | 33  | 33 * 8B = 264 bytes | 32-64 KB |
| 32   | 65  | 65 * 8B = 520 bytes | 32-64 KB |
| 64   | 129 | 129 * 8B = 1 KB     | 32-64 KB |

Even at band=64, the working set per elimination step (~1 KB for the pivot
row + subordinate rows) is **50-60x smaller** than L1 cache. There is no
cache miss to optimize away.

For dense matrices, the working set is N * sizeof(double) per column
(~800 KB for N=100,000), which far exceeds L1/L2 cache. That's where
blocking helps.

### 2. The Point Algorithm Is Already Optimal

The NR3 point algorithm (`bandec`) has these properties:
- **Sequential row access**: each elimination step reads the pivot row
  and m1 adjacent subordinate rows — all contiguous in memory
- **Fixed stride**: compact storage with stride = m1 + m2 + 1
- **No random access**: every memory access is predictable by the
  hardware prefetcher
- **O(1) overhead per element**: no index lookups, no bounds checks

These are exactly the properties that make cache blocking unnecessary.

### 3. Blocking Adds Overhead

The blocked algorithm introduces costs that don't exist in the point version:
- **Bounds checking**: LAPACK-style storage requires validity checks for
  each (i,j) access near matrix boundaries
- **Index computation**: the non-shifted storage requires computing
  `m1 + j - i` for every access (vs NR3's simpler shift-based indexing)
- **DTRSM step**: the triangular solve for panel rows is pure overhead —
  the point algorithm handles this implicitly
- **Triple-nested loops**: the DGEMM has 3 loop levels vs the point
  algorithm's 2

### 4. The DGEMM Is Too Small

For the DGEMM to provide cache benefit, it needs to be large enough
that the inner dimension (NB) provides meaningful data reuse:

| Band | DGEMM dimensions  | FLOPS | vs point overhead |
|------|-------------------|-------|-------------------|
| 3    | 3 x 3 x 3         | 27    | not worth it      |
| 16   | 16 x 16 x 32      | 8K    | marginal          |
| 32   | 32 x 32 x 32      | 32K   | still too small   |
| 64   | 64 x 64 x 32      | 131K  | could help if optimized |

At band=64, the DGEMM is ~131K FLOPS with NB=32. In a well-tuned BLAS
library, this could run at near-peak FLOPS. But our implementation uses
scalar C++ loops, not a tuned BLAS kernel. And the point algorithm already
runs at high efficiency because the data fits in cache anyway.

## When Would Blocking Help?

Blocking would become beneficial for banded matrices when:

1. **Bandwidth > ~1000**: working set exceeds L1 cache
2. **Tuned BLAS library available**: the DGEMM uses SIMD-optimized kernels
   (Intel MKL, OpenBLAS, etc.) that achieve near-peak throughput
3. **Column-major storage**: Fortran-style storage aligns with DTRSM/DGEMM
   access patterns

At bandwidth > 1000, the matrix is effectively dense and a dense solver
(with its mature blocking infrastructure) would be more appropriate anyway.

## Conclusion

The LAPACK block algorithm was a **historic breakthrough for dense matrices**,
enabling near-peak FLOPS by restructuring operations to use cache-optimal
BLAS-3 kernels. However, for banded matrices:

- The compact storage already ensures **L1 cache residency**
- The point algorithm's sequential access pattern is already **prefetcher-friendly**
- Blocking adds **overhead without cache benefit**
- The NR3 point algorithm with our micro-optimizations (branch removal,
  reciprocal precomputation) is the optimal approach

**The right optimization for banded matrices is not blocking — it's the
banded storage itself.** The O(N * bandwidth) algorithm with compact contiguous
storage is the LAPACK-equivalent "milestone" for the banded case: it reduces
the problem from cache-hostile O(N^3) to cache-friendly O(N * bw), making
blocking unnecessary.

## Files

| File                        | Description                              |
|-----------------------------|------------------------------------------|
| `src/banded_matrix.h`      | BandMatrixBlocked struct and API          |
| `src/banded_matrix.cpp`    | Blocked LU implementation                |
| `test/bench_blocked.cpp`   | Point vs blocked benchmark               |
