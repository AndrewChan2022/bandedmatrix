#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cassert>

namespace banded {

// Compact band matrix storage following Numerical Recipes Ch 2.4.
// For an n x n matrix with m1 lower diagonals and m2 upper diagonals,
// element A[i][j] is stored only when j in [i-m1, i+m2].
//
// Compact storage (flat, row-major, contiguous for data locality):
//   au: n rows x mm columns, where mm = m1 + m2 + 1
//   au[i * mm + (j - i + m1)] = A[i][j]
//   So the diagonal A[i][i] is at column m1 in each row.
//
// After LU decomposition (bandec):
//   au is rearranged so diagonal is at column 0 of each row.
//   al: separate n x m1 storage for L multipliers.
//
// Memory: O(N * bandwidth). LU solve: O(N * bandwidth^2).

struct BandMatrix {
    int n;       // matrix dimension
    int m1;      // number of sub-diagonals (lower bandwidth)
    int m2;      // number of super-diagonals (upper bandwidth)
    int mm;      // band width = m1 + m2 + 1

    // Compact storage for U (and original matrix before decomposition).
    // Flat array, n * mm elements, row-major.
    std::vector<double> au;

    // Lower triangular factors from LU decomposition.
    // Flat array, n * m1 elements.
    std::vector<double> al;

    // Precomputed reciprocals of U diagonal (1.0 / au[k*mm+0]) for back-substitution.
    // Avoids division in the solve loop (multiply is ~4x faster than divide).
    std::vector<double> diag_inv;

    // Pivot indices for LU decomposition
    std::vector<int> pivot;

    // Determinant sign (+1 or -1) from row swaps
    double d;

    BandMatrix() : n(0), m1(0), m2(0), mm(0), d(1.0) {}

    BandMatrix(int n_, int m1_, int m2_)
        : n(n_), m1(m1_), m2(m2_), mm(m1_ + m2_ + 1), d(1.0)
    {
        au.assign(static_cast<size_t>(n) * mm, 0.0);
        al.assign(static_cast<size_t>(n) * m1, 0.0);
        diag_inv.resize(n, 0.0);
        pivot.resize(n, 0);
    }

    // Access element A[i][j] in the original (pre-decomposition) compact storage.
    double& operator()(int i, int j) {
        return au[static_cast<size_t>(i) * mm + (j - i + m1)];
    }

    double operator()(int i, int j) const {
        return au[static_cast<size_t>(i) * mm + (j - i + m1)];
    }
};

// --------------------------------------------------------------------------
// LU Decomposition for Band Matrices (Numerical Recipes Ch 2.4, bandec)
// --------------------------------------------------------------------------

// In-place LU decomposition of a band matrix.
// After this call, au contains U (rearranged), al contains L multipliers.
// Returns false if matrix is singular.
bool band_lu_decompose(BandMatrix& a);

// Solve A*x = b using previously LU-decomposed band matrix.
// b is overwritten with the solution x.
void band_lu_solve(const BandMatrix& a, double* b);

// Solve A*x = b (convenience: allocates result).
std::vector<double> band_lu_solve(const BandMatrix& a, const std::vector<double>& b);

// --------------------------------------------------------------------------
// Blocked LU Decomposition (LAPACK-style, BLAS-3 trailing update)
// --------------------------------------------------------------------------
// Uses panel factorization + DGEMM-like trailing update for better
// cache utilization at larger bandwidths. Based on the LAPACK dgbtrf
// milestone: reorganize elimination to use Level-3 BLAS operations.
//
// Internal storage: LAPACK-style, ldab = 2*m1 + m2 + 1, no row-shift.
// Element A(i,j) at ab[i * ldab + (m1 + j - i)], diagonal at column m1.
// Extra m1 columns (0..m1-1) for fill-in from row pivoting.

struct BandMatrixBlocked {
    int n;
    int m1;
    int m2;
    int ldab;    // = 2*m1 + m2 + 1

    std::vector<double> ab;       // n * ldab, LAPACK-style band storage
    std::vector<double> diag_inv; // precomputed 1/diagonal
    std::vector<int> pivot;
    double d;

    BandMatrixBlocked() : n(0), m1(0), m2(0), ldab(0), d(1.0) {}

    BandMatrixBlocked(int n_, int m1_, int m2_)
        : n(n_), m1(m1_), m2(m2_), ldab(2 * m1_ + m2_ + 1), d(1.0)
    {
        ab.assign(static_cast<size_t>(n) * ldab, 0.0);
        diag_inv.resize(n, 0.0);
        pivot.resize(n, 0);
    }

    // Access A(i,j) in LAPACK-style storage (before decomposition)
    double& operator()(int i, int j) {
        return ab[static_cast<size_t>(i) * ldab + (m1 + j - i)];
    }
    double operator()(int i, int j) const {
        return ab[static_cast<size_t>(i) * ldab + (m1 + j - i)];
    }
};

// Blocked LU decomposition. nb = block size (default 32).
bool band_lu_decompose_blocked(BandMatrixBlocked& a, int nb = 32);

// Solve using blocked LU decomposition result.
void band_lu_solve_blocked(const BandMatrixBlocked& a, double* b);
std::vector<double> band_lu_solve_blocked(const BandMatrixBlocked& a, const std::vector<double>& b);

// --------------------------------------------------------------------------
// QR Decomposition for Band Matrices
// --------------------------------------------------------------------------
// Uses Givens rotations to exploit banded structure.
// R has upper bandwidth m1+m2 (fill-in from rotations).

struct BandQR {
    int n;
    int m1;
    int m2;
    int stride;       // stride for R storage: m1 + m2 + 1
    std::vector<double> r_data; // R in compact band storage
    std::vector<double> cs;    // cosines of Givens rotations
    std::vector<double> sn;    // sines of Givens rotations
    int n_rotations;

    BandQR() : n(0), m1(0), m2(0), stride(0), n_rotations(0) {}
};

// Compute QR decomposition of band matrix using Givens rotations.
BandQR band_qr_decompose(const BandMatrix& a);

// Solve A*x = b using QR decomposition: x = R^{-1} * Q^T * b.
void band_qr_solve(const BandQR& qr, double* b);

// Convenience wrapper
std::vector<double> band_qr_solve(const BandQR& qr, const std::vector<double>& b);

} // namespace banded
