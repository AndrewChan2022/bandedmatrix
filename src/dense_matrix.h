#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace dense {

// Dense matrix stored as flat contiguous array (row-major) for data locality.
// No vector-of-vector.
struct DenseMatrix {
    int n;
    std::vector<double> data; // n * n elements, row-major
    std::vector<int> pivot;
    double d; // determinant sign

    DenseMatrix() : n(0), d(1.0) {}

    DenseMatrix(int n_) : n(n_), d(1.0) {
        data.assign(static_cast<size_t>(n) * n, 0.0);
        pivot.resize(n, 0);
    }

    double& operator()(int i, int j) { return data[static_cast<size_t>(i) * n + j]; }
    double operator()(int i, int j) const { return data[static_cast<size_t>(i) * n + j]; }
};

// --------------------------------------------------------------------------
// Dense LU Decomposition (Crout's method with partial pivoting, NR Ch 2.3)
// --------------------------------------------------------------------------

bool dense_lu_decompose(DenseMatrix& a);
void dense_lu_solve(const DenseMatrix& a, double* b);
std::vector<double> dense_lu_solve(const DenseMatrix& a, const std::vector<double>& b);

// --------------------------------------------------------------------------
// Dense QR Decomposition (Householder reflections, NR Ch 2.10)
// --------------------------------------------------------------------------

struct DenseQR {
    int n;
    std::vector<double> qt_data; // Q^T stored as n*n flat array
    std::vector<double> r_data;  // R stored as n*n flat array (upper triangular)

    DenseQR() : n(0) {}
};

DenseQR dense_qr_decompose(const DenseMatrix& a);
void dense_qr_solve(const DenseQR& qr, double* b);
std::vector<double> dense_qr_solve(const DenseQR& qr, const std::vector<double>& b);

} // namespace dense
