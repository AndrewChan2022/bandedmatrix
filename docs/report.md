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

1. **Data locality**: All matrices stored as flat contiguous arrays (no vector-of-vector).
   Band matrices use compact storage with stride = m1 + m2 + 1.
2. **Minimal branching**: Inner loops avoid conditionals where possible.
3. **Efficient algorithms**: Banded solvers exploit structure for O(N * bandwidth) time
   vs O(N^3) for dense, yielding orders-of-magnitude speedup.
4. **Cache-friendly access**: Row-major layout with sequential access patterns.

## Benchmark Results

Environment: Windows 11, MSVC 19.44 (VS2022), /O2 optimization, single-threaded.
Eigen version: 3.4.90.

All matrices are diagonally dominant random banded matrices with band = 3
(m1 = m2 = 3, bandwidth = 7).

### (a) Small Matrix: N = 10

| Method           | Decomp (ms) | Solve (ms) | Total (ms) | Residual |
|------------------|-------------|------------|------------|----------|
| Our Banded LU    | 0.001       | 0.004      | 0.005      | 1.8e-15  |
| Our Banded QR    | 0.008       | 0.001      | 0.008      | 3.6e-15  |
| Our Dense LU     | 0.001       | 0.005      | 0.006      | 1.8e-15  |
| Our Dense QR     | 0.004       | 0.004      | 0.008      | 6.2e-15  |
| Eigen Dense LU   | 0.028       | 0.018      | 0.047      | 1.8e-15  |
| Eigen Dense QR   | 0.037       | 0.014      | 0.051      | 2.9e-15  |
| Eigen Sparse LU  | 0.064       | 0.005      | 0.069      | 3.6e-15  |
| Eigen Sparse QR  | 0.013       | 0.008      | 0.021      | 5.3e-15  |

At small N, all methods are essentially instantaneous. Eigen has higher overhead
from object construction and SIMD setup. All residuals at machine precision.

### (b) Middle Scale: N = 1,000

| Method           | Decomp (ms) | Solve (ms) | Total (ms) | Residual |
|------------------|-------------|------------|------------|----------|
| Our Banded LU    | 0.016       | 0.018      | 0.034      | 5.3e-15  |
| Our Banded QR    | 0.152       | 0.022      | 0.174      | 7.1e-15  |
| Our Dense LU     | 161.6       | 1.7        | 163.3      | 5.3e-15  |
| Our Dense QR     | 2069.1      | 1.5        | 2070.5     | 1.1e-14  |
| Eigen Dense LU   | 50.8        | 0.4        | 51.2       | 5.3e-15  |
| Eigen Dense QR   | 119.1       | 0.7        | 119.8      | 7.1e-15  |
| Eigen Sparse LU  | 0.852       | 0.029      | 0.881      | 5.3e-15  |
| Eigen Sparse QR  | 1.014       | 0.064      | 1.078      | 8.9e-15  |

Key observations at N = 1,000:
- **Our Banded LU is 1,500x faster than Eigen Dense LU** (0.034 vs 51.2 ms)
- **Our Banded LU is 26x faster than Eigen Sparse LU** (0.034 vs 0.881 ms)
- Eigen Dense LU is 3.2x faster than Our Dense LU (SIMD advantage)
- All LU methods produce identical residuals (5.3e-15)

### (d) Large Scale: N = 1,000,000

Dense solvers are infeasible at this scale (N^2 = 10^12 doubles = 8 TB RAM).

| Method           | Decomp (ms) | Solve (ms) | Total (ms) | Residual |
|------------------|-------------|------------|------------|----------|
| Our Banded LU    | 27.5        | 24.4       | 52.0       | 8.9e-15  |
| Our Banded QR    | 164.9       | 26.0       | 190.9      | 1.3e-14  |
| Eigen Sparse LU  | 1026.4      | 32.0       | 1058.4     | 1.1e-14  |
| Eigen Sparse QR  | 426855.9    | 51.7       | 426907.6   | 1.4e-14  |

Key observations at N = 1,000,000:
- **Our Banded LU is 20x faster than Eigen Sparse LU** (52 ms vs 1,058 ms)
- **Our Banded LU is 8,200x faster than Eigen Sparse QR** (52 ms vs 426,908 ms)
- Eigen Sparse QR's COLAMD ordering analysis is extremely expensive at this scale
  (426 seconds just for decomposition)
- All methods achieve machine-precision residuals

## Performance Comparison Summary

### Total Solve Time (ms) — LU Methods

| N         | Our Banded LU | Eigen Sparse LU | Eigen Dense LU | Our Dense LU |
|-----------|---------------|-----------------|----------------|--------------|
| 10        | 0.005         | 0.069           | 0.047          | 0.006        |
| 1,000     | 0.034         | 0.881           | 51.2           | 163.3        |
| 1,000,000 | 52.0          | 1,058.4         | N/A (8 TB)     | N/A (8 TB)   |

### Speedup vs Eigen Sparse LU

| N         | Our Banded LU (ms) | Eigen Sparse LU (ms) | Speedup |
|-----------|---------------------|-----------------------|---------|
| 10        | 0.005               | 0.069                 | 14x     |
| 1,000     | 0.034               | 0.881                 | 26x     |
| 1,000,000 | 52.0                | 1,058.4               | 20x     |

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
| `demo/demo.cpp`               | Demo and performance showcase            |
| `CMakeLists.txt`              | CMake build (cmake + VS2022)             |

## Conclusion

Our banded matrix solver, based on Numerical Recipes Chapter 2.4, provides:

- **20x faster than Eigen Sparse LU** at N = 1,000,000 (52 ms vs 1,058 ms)
- **1,500x faster than Eigen Dense LU** at N = 1,000 (0.034 ms vs 51.2 ms)
- **Identical precision** to Eigen Dense LU (both achieve 5.3e-15 residual at N=1,000)
- **O(N * bandwidth^2) complexity** — practically O(N) for small bandwidths
- **1 million unknowns solved in 52 ms** on a single thread

The key insight: when the problem structure is known (banded matrix), a specialized
solver with compact contiguous storage vastly outperforms general-purpose sparse or
dense solvers, in both speed and memory usage.
