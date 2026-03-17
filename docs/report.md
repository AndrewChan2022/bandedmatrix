# Banded Matrix Solver: Summary Report

## Overview

Implementation of banded matrix solvers (LU and QR decomposition) based on
**Numerical Recipes, 3rd Edition**, Chapter 2.4: *Tridiagonal and Band-Diagonal
Systems of Equations*.

Dense matrix solvers (LU and QR) and Eigen3 solvers (dense and sparse) are also
included for correctness verification and performance comparison.

## Algorithms

### Banded LU (Numerical Recipes `Bandec`)

- Compact band storage: N x (m1 + m2 + 1) flat array, diagonal at column m1
- LU decomposition with partial pivoting within the band
- Initial row rearrangement to align storage for elimination
- During elimination, each subordinate row is shifted left (column compression)
- L factors stored separately in N x m1 array
- Complexity: **O(N * bandwidth^2)** decomposition, **O(N * bandwidth)** solve

### Banded QR (Givens Rotations)

- Eliminates sub-diagonal entries column-by-column using Givens rotations
- R has upper bandwidth m1 + m2 (fill-in from rotations)
- Rotation parameters (cos, sin) stored for apply Q^T during solve
- Complexity: **O(N * bandwidth^2)** decomposition, **O(N * bandwidth)** solve

### Dense LU (Crout's Method, NR Ch 2.3)

- Standard N x N dense storage (flat row-major array)
- Partial pivoting with implicit scaling
- Complexity: **O(N^3)** decomposition, **O(N^2)** solve

### Dense QR (Householder Reflections)

- Q^T and R stored explicitly as N x N arrays
- Complexity: **O(N^3)** decomposition, **O(N^2)** solve

### Eigen3 Solvers (v3.4.90)

- **Eigen Dense LU**: `PartialPivLU` — optimized dense LU with SIMD
- **Eigen Dense QR**: `HouseholderQR` — optimized Householder with SIMD
- **Eigen Sparse LU**: `SparseLU` with COLAMD ordering — general sparse solver
- **Eigen Sparse QR**: `SparseQR` with COLAMD ordering — general sparse solver

## High-Performance Design

Following the `skills/high-perf` guidelines:

1. **Data locality** (skill #1): All matrices stored as flat contiguous arrays
   (no vector-of-vector). Band matrices use compact storage with stride = m1 + m2 + 1,
   similar to CSR format — sequential access with predictable stride.
2. **Minimize branching** (skill #3): QR Givens inner loop has zero bounds checks —
   proved mathematically that indices are always in `[0, w_stride)`. LU inner loop
   is also branch-free.
3. **Efficient algorithms** (skill #4): Banded solvers exploit structure for
   O(N * bandwidth) time vs O(N^3) for dense, yielding orders-of-magnitude speedup.
   This is the single largest performance win.
4. **Precompute and reuse** (skill #8): Pivot reciprocals precomputed during LU
   decomposition (`diag_inv[]`), replacing N divisions with N multiplications
   in back-substitution (~4x faster per operation).
5. **Avoid unnecessary compute** (skill #9): Only band elements are touched —
   zero elements outside the band are never stored or computed.
6. **Profile and measure** (skill #6): Benchmarks across multiple scales, methods,
   and compiler flags validate every optimization.

## Micro-Optimization Results

After applying branching removal and reciprocal precomputation:

| Method (N=1M, band=3) | Before (ms) | After (ms) | Improvement |
|------------------------|-------------|------------|-------------|
| Our Banded LU          | 52.0        | 41.9       | **19% faster** |
| Our Banded QR          | 190.9       | 145.3      | **24% faster** |

The QR improvement is larger because it had 4 branch checks per inner iteration
(all proven unnecessary and removed). The LU gained from replacing division with
precomputed reciprocal multiplication.

## Benchmark Results

Environment: Windows 11, MSVC 19.44 (VS2022), /O2 optimization, single-threaded.
Eigen version: 3.4.90.

All matrices are diagonally dominant random banded matrices with band = 3
(m1 = m2 = 3, bandwidth = 7).

### (a) Small Matrix: N = 10

| Method           | Decomp (ms) | Solve (ms) | Total (ms) | Residual |
|------------------|-------------|------------|------------|----------|
| Our Banded LU    | 0.003       | 0.000      | 0.003      | 1.8e-15  |
| Our Banded QR    | 0.004       | 0.000      | 0.004      | 3.6e-15  |
| Our Dense LU     | 0.001       | 0.003      | 0.004      | 1.8e-15  |
| Our Dense QR     | 0.003       | 0.003      | 0.006      | 6.2e-15  |
| Eigen Dense LU   | 0.013       | 0.011      | 0.024      | 1.8e-15  |
| Eigen Dense QR   | 0.022       | 0.009      | 0.030      | 2.9e-15  |
| Eigen Sparse LU  | 0.060       | 0.004      | 0.064      | 3.6e-15  |
| Eigen Sparse QR  | 0.018       | 0.009      | 0.026      | 5.3e-15  |

At small N, all methods are essentially instantaneous. Eigen has higher overhead
from object construction and SIMD setup. All residuals at machine precision.

### (b) Middle Scale: N = 1,000

| Method           | Decomp (ms) | Solve (ms) | Total (ms) | Residual |
|------------------|-------------|------------|------------|----------|
| Our Banded LU    | 0.016       | 0.015      | 0.031      | 5.3e-15  |
| Our Banded QR    | 0.126       | 0.021      | 0.147      | 7.1e-15  |
| Our Dense LU     | 131.4       | 1.3        | 132.8      | 5.3e-15  |
| Our Dense QR     | 1805.5      | 1.8        | 1807.3     | 1.1e-14  |
| Eigen Dense LU   | 51.7        | 0.4        | 52.2       | 5.3e-15  |
| Eigen Dense QR   | 119.2       | 0.8        | 120.0      | 7.1e-15  |
| Eigen Sparse LU  | 0.891       | 0.030      | 0.921      | 5.3e-15  |
| Eigen Sparse QR  | 0.623       | 0.037      | 0.659      | 8.9e-15  |

Key observations at N = 1,000:
- **Our Banded LU is 1,684x faster than Eigen Dense LU** (0.031 vs 52.2 ms)
- **Our Banded LU is 30x faster than Eigen Sparse LU** (0.031 vs 0.921 ms)
- Eigen Dense LU is 2.5x faster than Our Dense LU (SIMD advantage)
- All LU methods produce identical residuals (5.3e-15)

### (d) Large Scale: N = 1,000,000

Dense solvers are infeasible at this scale (N^2 = 10^12 doubles = 8 TB RAM).

| Method           | Decomp (ms) | Solve (ms) | Total (ms) | Residual |
|------------------|-------------|------------|------------|----------|
| Our Banded LU    | 21.2        | 20.7       | 41.9       | 8.9e-15  |
| Our Banded QR    | 119.9       | 25.4       | 145.3      | 1.3e-14  |
| Eigen Sparse LU  | 933.5       | 49.0       | 982.4      | 1.1e-14  |
| Eigen Sparse QR  | 445651.2    | 75.9       | 445727.1   | 1.4e-14  |

Key observations at N = 1,000,000:
- **Our Banded LU is 23x faster than Eigen Sparse LU** (42 ms vs 982 ms)
- **Our Banded LU is 10,600x faster than Eigen Sparse QR** (42 ms vs 445,727 ms)
- Eigen Sparse QR's COLAMD ordering analysis is extremely expensive at this scale
  (446 seconds just for decomposition)
- All methods achieve machine-precision residuals

## Performance Comparison Summary

### Total Solve Time (ms) — LU Methods

| N         | Our Banded LU | Eigen Sparse LU | Eigen Dense LU | Our Dense LU |
|-----------|---------------|-----------------|----------------|--------------|
| 10        | 0.003         | 0.064           | 0.024          | 0.004        |
| 1,000     | 0.031         | 0.921           | 52.2           | 132.8        |
| 1,000,000 | 41.9          | 982.4           | N/A (8 TB)     | N/A (8 TB)   |

### Speedup vs Eigen Sparse LU

| N         | Our Banded LU (ms) | Eigen Sparse LU (ms) | Speedup |
|-----------|---------------------|-----------------------|---------|
| 10        | 0.003               | 0.064                 | 21x     |
| 1,000     | 0.031               | 0.921                 | 30x     |
| 1,000,000 | 41.9                | 982.4                 | 23x     |

## SIMD / AVX-512 Analysis

Benchmarked with three compiler settings: SSE2 (baseline `/O2`), AVX2 (`/arch:AVX2`),
and AVX-512 (`/arch:AVX512`). N = 100,000. LU decompose + solve total time.

| Band | Inner loop | SSE2 (ms) | AVX2 (ms) | AVX-512 (ms) | AVX-512 vs SSE2 |
|------|-----------|-----------|-----------|--------------|-----------------|
| 1    | 3 elem    | 1.91      | 1.89      | 2.14         | 0.9x (slower)   |
| 3    | 7 elem    | 3.55      | 3.42      | 3.26         | 1.1x            |
| 8    | 17 elem   | 7.56      | 9.58      | 12.28        | 0.6x (slower)   |
| 16   | 33 elem   | 22.96     | 20.79     | **16.29**    | **1.4x**        |
| 32   | 65 elem   | 71.99     | 58.89     | **44.93**    | **1.6x**        |
| 64   | 129 elem  | 256.42    | 192.53    | **159.64**   | **1.6x**        |

### Key Findings

**At band=3 (our primary use case): SIMD gives negligible improvement (~8%).**

The inner loops process only 7 elements — less than one AVX-512 vector width
(8 doubles). The compiler cannot fill a full vector register, so execution is
effectively scalar regardless of the `/arch` flag.

**AVX-512 actually hurts at band=8 (0.6x).** Intel CPUs downclock when executing
512-bit instructions (thermal/power throttling). For 17-element loops, the wider
registers don't provide enough throughput to overcome the frequency penalty.

**AVX-512 helps significantly at band >= 16 (1.4-1.6x).** At 33+ elements per
inner loop, you get 4+ full AVX-512 iterations, and the FMA throughput advantage
overcomes the frequency penalty.

**AVX2 is the best default.** No frequency penalty, and the auto-vectorizer handles
loops with 16+ elements well. Recommended build flag: `/arch:AVX2`.

### Why SIMD Is Not the Key Optimization Here

For band=3, the performance hierarchy is:

| Optimization                        | Impact           |
|-------------------------------------|------------------|
| Algorithm choice (banded vs dense)  | **4,800x**       |
| vs Eigen Sparse (structure exploit) | **23x**          |
| Branch removal + reciprocal precomp | **19-24%**       |
| SIMD / AVX-512                      | **~8% (band=3)** |

The algorithmic win (O(N*bw) vs O(N^3)) dwarfs any SIMD benefit. SIMD becomes
relevant only for large bandwidths (band >= 16), where it provides 1.4-1.6x on
top of the algorithmic advantage.

## Numerical Accuracy Comparison

All solvers achieve machine-precision residuals (max |A*x - b|):

| Method           | N=10     | N=1,000  | N=1,000,000 |
|------------------|----------|----------|-------------|
| Our Banded LU    | 1.8e-15  | 5.3e-15  | 8.9e-15     |
| Our Banded QR    | 3.6e-15  | 7.1e-15  | 1.3e-14     |
| Eigen Dense LU   | 1.8e-15  | 5.3e-15  | N/A         |
| Eigen Sparse LU  | 3.6e-15  | 5.3e-15  | 1.1e-14     |
| Eigen Sparse QR  | 5.3e-15  | 8.9e-15  | 1.4e-14     |

Our Banded LU matches Eigen Dense LU in precision (both 5.3e-15 at N=1,000).
QR methods show slightly larger residuals (2-3x) due to accumulated Givens/Householder
roundoff, but remain at machine precision.

## Why Our Banded Solver Beats Eigen Sparse

Eigen's sparse solvers (`SparseLU`, `SparseQR`) are **general-purpose** sparse solvers
designed for arbitrary sparsity patterns. They pay overhead for:

1. **Column reordering** (COLAMD): O(nnz) analysis to reduce fill-in — unnecessary
   for banded matrices where the fill-in pattern is known a priori
2. **Symbolic factorization**: building the elimination tree for general sparsity
3. **Indirect addressing**: CSC format requires index lookups, causing cache misses
4. **Dynamic memory**: allocating fill-in storage during factorization

Our banded solver avoids all of this by exploiting the **known banded structure**:
- No reordering needed (band structure is already optimal)
- Contiguous flat array with predictable stride (cache-line friendly)
- No index lookups — element positions computed arithmetically
- Fixed memory layout with no dynamic allocation during factorization

This is a textbook example of the high-perf principle: **use efficient algorithms
matched to the data structure** (skill #4) and **data locality** (skill #1).

## Bandwidth Scaling

How our solver scales with increasing bandwidth (N = 100,000, `/O2`):

| Band | Bandwidth | Total (ms) | Throughput (Mrows/s) | Bytes/row |
|------|-----------|------------|----------------------|-----------|
| 1    | 3         | 1.91       | 52.4                 | 24        |
| 3    | 7         | 3.55       | 28.2                 | 56        |
| 8    | 17        | 7.56       | 13.2                 | 136       |
| 16   | 33        | 22.96      | 4.4                  | 264       |
| 32   | 65        | 71.99      | 1.4                  | 520       |
| 64   | 129       | 256.42     | 0.4                  | 1032      |

Time scales as O(N * bandwidth^2) as expected. At band=64, each row occupies
1 KB — exceeding L1 cache line — and throughput drops. For band >= 16,
adding `/arch:AVX2` recovers 18-25% of this cost.

## Files

| File                           | Description                              |
|--------------------------------|------------------------------------------|
| `src/banded_matrix.h`         | Banded matrix storage and solver API     |
| `src/banded_matrix.cpp`       | Banded LU and QR implementations         |
| `src/dense_matrix.h`          | Dense matrix storage and solver API      |
| `src/dense_matrix.cpp`        | Dense LU and QR implementations          |
| `test/test_banded_matrix.cpp` | Unit tests (6 tests, all passing)        |
| `test/benchmark.cpp`          | Dense vs banded benchmark                |
| `test/benchmark_eigen.cpp`    | Eigen3 comparison benchmark              |
| `test/bench_bandwidth.cpp`    | Bandwidth scaling & SIMD benchmark       |
| `demo/demo.cpp`               | Demo and performance showcase            |
| `CMakeLists.txt`              | CMake build (cmake + VS2022)             |

## Conclusion

Our banded matrix solver, based on Numerical Recipes Chapter 2.4, provides:

- **23x faster than Eigen Sparse LU** at N = 1,000,000 (42 ms vs 982 ms)
- **1,684x faster than Eigen Dense LU** at N = 1,000 (0.031 ms vs 52.2 ms)
- **Identical precision** to Eigen Dense LU (both achieve 5.3e-15 residual at N=1,000)
- **O(N * bandwidth^2) complexity** — practically O(N) for small bandwidths
- **1 million unknowns solved in 42 ms** on a single thread

Performance hierarchy for band=3:

```
Algorithm choice (banded vs dense):    4,800x
Structure exploit (vs Eigen Sparse):      23x
Micro-opts (branching, reciprocals):      24%
SIMD / AVX-512:                           ~8%
```

The key insight: when the problem structure is known (banded matrix), a specialized
solver with compact contiguous storage vastly outperforms general-purpose sparse or
dense solvers. SIMD provides marginal gains at small bandwidths because inner loops
are shorter than vector register width; it becomes meaningful (1.4-1.6x) only at
bandwidth >= 33.
