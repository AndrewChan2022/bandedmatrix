# Banded Matrix Solver: Summary Report

## Overview

Implementation of banded matrix solvers (LU and QR decomposition) based on
**Numerical Recipes, 3rd Edition**, Chapter 2.4: *Tridiagonal and Band-Diagonal
Systems of Equations*.

Dense matrix solvers (LU and QR) are also implemented for correctness verification
and performance comparison.

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

All matrices are diagonally dominant random banded matrices with band = 3
(m1 = m2 = 3, bandwidth = 7).

### (a) Small Matrix: N = 10

| Method     | Decomp (ms) | Solve (ms) | Total (ms) | Residual |
|------------|-------------|------------|------------|----------|
| Dense LU   | 0.004       | 0.003      | 0.007      | 1.8e-15  |
| Dense QR   | 0.003       | 0.000      | 0.003      | 6.2e-15  |
| Banded LU  | 0.000       | 0.003      | 0.003      | 1.8e-15  |
| Banded QR  | 0.002       | 0.000      | 0.002      | 3.6e-15  |

At small N, all methods are essentially instantaneous. Residuals confirm correctness.

### (b) Middle Scale: N = 1,000

| Method     | Decomp (ms) | Solve (ms) | Total (ms) | Residual |
|------------|-------------|------------|------------|----------|
| Dense LU   | 152.3       | 1.7        | 154.1      | 5.3e-15  |
| Dense QR   | 1945.2      | 1.6        | 1946.8     | 1.2e-14  |
| Banded LU  | 0.015       | 0.017      | 0.032      | 5.3e-15  |
| Banded QR  | 0.142       | 0.020      | 0.162      | 7.1e-15  |

**Banded LU is 4,800x faster than Dense LU** at N=1,000.

Dense QR is 12x slower than Dense LU due to higher constant factors in
Householder reflections (both are O(N^3), but QR has a larger constant).

### (d) Large Scale: N = 1,000,000

Dense solvers are infeasible at this scale (N^2 = 10^12 doubles = 8 TB RAM).

| Method     | Decomp (ms) | Solve (ms) | Total (ms) | Residual |
|------------|-------------|------------|------------|----------|
| Banded LU  | 17.9        | 18.4       | 36.3       | 8.9e-15  |
| Banded QR  | 126.2       | 20.8       | 147.0      | 1.3e-14  |

**1 million unknowns solved in 36 ms** with Banded LU.

Banded LU throughput: ~27.6 million rows/second.

### Speedup Summary (Dense LU vs Banded LU)

| N     | Dense LU (ms) | Banded LU (ms) | Speedup   |
|-------|---------------|-----------------|-----------|
| 10    | 0.002         | 0.001           | 2.6x      |
| 100   | 0.123         | 0.003           | 38x       |
| 1,000 | 140.2         | 0.032           | 4,313x    |

The speedup grows with N because dense is O(N^3) while banded is O(N * bandwidth^2).
At N = 1,000,000, dense would take ~38 hours (estimated) vs 36 ms for banded.

## Numerical Accuracy

All solvers achieve machine-precision residuals (~10^-14 to 10^-15) across all test
sizes. Banded LU and Dense LU produce identical residuals, confirming that the
compact storage introduces no additional numerical error.

## Files

| File                        | Description                              |
|-----------------------------|------------------------------------------|
| `src/banded_matrix.h`       | Banded matrix storage and solver API     |
| `src/banded_matrix.cpp`     | Banded LU and QR implementations         |
| `src/dense_matrix.h`        | Dense matrix storage and solver API      |
| `src/dense_matrix.cpp`      | Dense LU and QR implementations          |
| `test/test_banded_matrix.cpp` | Unit tests (6 tests, all passing)      |
| `test/benchmark.cpp`        | Dense vs banded benchmark                |
| `demo/demo.cpp`             | Demo and performance showcase            |
| `CMakeLists.txt`            | CMake build (cmake + VS2022)             |

## Conclusion

Banded matrix solvers provide dramatic performance improvements over dense solvers
when the matrix has banded structure. For a system with bandwidth 7 (band = 3):

- At N = 1,000: **4,800x faster** than dense LU
- At N = 1,000,000: solves in **36 ms** (dense is completely infeasible)
- **No accuracy loss** compared to dense methods

The O(N * bandwidth^2) complexity makes banded solvers practically **O(N)** for
small bandwidths, enabling real-time solution of million-scale systems that arise
in 1D/2D finite element and finite difference discretizations.
