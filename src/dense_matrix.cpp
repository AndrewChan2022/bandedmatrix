#include "dense_matrix.h"
#include <cmath>
#include <algorithm>
#include <utility>
#include <stdexcept>

namespace dense {

// --------------------------------------------------------------------------
// Dense LU Decomposition (Crout's method, NR Ch 2.3)
// --------------------------------------------------------------------------

bool dense_lu_decompose(DenseMatrix& a) {
    const int n = a.n;
    double* const m = a.data.data();

    a.d = 1.0;

    // Implicit scaling for each row (for partial pivoting)
    std::vector<double> vv(n);
    for (int i = 0; i < n; i++) {
        double big = 0.0;
        for (int j = 0; j < n; j++) {
            double temp = std::abs(m[static_cast<size_t>(i) * n + j]);
            if (temp > big) big = temp;
        }
        if (big == 0.0) return false; // singular
        vv[i] = 1.0 / big;
    }

    for (int k = 0; k < n; k++) {
        // Find pivot
        double big = 0.0;
        int imax = k;
        for (int i = k; i < n; i++) {
            double temp = vv[i] * std::abs(m[static_cast<size_t>(i) * n + k]);
            if (temp > big) {
                big = temp;
                imax = i;
            }
        }

        // Swap rows
        if (imax != k) {
            for (int j = 0; j < n; j++) {
                std::swap(m[static_cast<size_t>(imax) * n + j],
                          m[static_cast<size_t>(k) * n + j]);
            }
            a.d = -a.d;
            vv[imax] = vv[k];
        }
        a.pivot[k] = imax;

        if (m[static_cast<size_t>(k) * n + k] == 0.0) return false;

        // Eliminate
        double inv_pivot = 1.0 / m[static_cast<size_t>(k) * n + k];
        for (int i = k + 1; i < n; i++) {
            double factor = m[static_cast<size_t>(i) * n + k] * inv_pivot;
            m[static_cast<size_t>(i) * n + k] = factor;
            for (int j = k + 1; j < n; j++) {
                m[static_cast<size_t>(i) * n + j] -= factor * m[static_cast<size_t>(k) * n + j];
            }
        }
    }
    return true;
}

void dense_lu_solve(const DenseMatrix& a, double* b) {
    const int n = a.n;
    const double* m = a.data.data();

    // Forward substitution
    for (int k = 0; k < n; k++) {
        if (a.pivot[k] != k) std::swap(b[k], b[a.pivot[k]]);
        for (int i = k + 1; i < n; i++) {
            b[i] -= m[static_cast<size_t>(i) * n + k] * b[k];
        }
    }

    // Back substitution
    for (int k = n - 1; k >= 0; k--) {
        double sum = b[k];
        for (int j = k + 1; j < n; j++) {
            sum -= m[static_cast<size_t>(k) * n + j] * b[j];
        }
        b[k] = sum / m[static_cast<size_t>(k) * n + k];
    }
}

std::vector<double> dense_lu_solve(const DenseMatrix& a, const std::vector<double>& b) {
    if (static_cast<int>(b.size()) != a.n)
        throw std::invalid_argument("RHS size mismatch");
    std::vector<double> x = b;
    dense_lu_solve(a, x.data());
    return x;
}

// --------------------------------------------------------------------------
// Blocked Dense LU (LAPACK-style, right-looking)
// --------------------------------------------------------------------------
// For each block of NB columns:
//   1. Panel factorization: point LU on NB columns (BLAS-2)
//   2. DTRSM: compute U12 = L11^{-1} * A12 for panel rows, trailing cols
//   3. DGEMM: A22 -= L21 * U12 for below-panel rows, trailing cols
//
// The DGEMM in step 3 reuses each element ~NB times in cache, converting
// memory-bound BLAS-2 into compute-bound BLAS-3.

bool dense_lu_decompose_blocked(DenseMatrix& a, int nb) {
    const int n = a.n;
    double* m = a.data.data();

    a.d = 1.0;
    if (nb < 1) nb = 1;
    if (nb > n) nb = n;

    // Implicit scaling for pivoting
    std::vector<double> vv(n);
    for (int i = 0; i < n; i++) {
        double big = 0.0;
        for (int j = 0; j < n; j++) {
            double temp = std::abs(m[static_cast<size_t>(i) * n + j]);
            if (temp > big) big = temp;
        }
        if (big == 0.0) return false;
        vv[i] = 1.0 / big;
    }

    for (int jb = 0; jb < n; jb += nb) {
        int jend = std::min(jb + nb, n);

        // ---- Step 1: Panel factorization (columns jb..jend-1) ----
        // Point algorithm, but only update within-panel columns.
        for (int k = jb; k < jend; k++) {
            // Find pivot
            double big = 0.0;
            int imax = k;
            for (int i = k; i < n; i++) {
                double temp = vv[i] * std::abs(m[static_cast<size_t>(i) * n + k]);
                if (temp > big) {
                    big = temp;
                    imax = i;
                }
            }

            if (imax != k) {
                // Swap entire rows (all n columns, not just panel)
                // Need full swap so that trailing columns are correct for DTRSM
                for (int j = 0; j < n; j++) {
                    std::swap(m[static_cast<size_t>(imax) * n + j],
                              m[static_cast<size_t>(k) * n + j]);
                }
                a.d = -a.d;
                vv[imax] = vv[k];
            }
            a.pivot[k] = imax;

            double diag = m[static_cast<size_t>(k) * n + k];
            if (diag == 0.0) return false;

            double inv_pivot = 1.0 / diag;

            // Compute L factors and update within-panel columns only
            for (int i = k + 1; i < n; i++) {
                double factor = m[static_cast<size_t>(i) * n + k] * inv_pivot;
                m[static_cast<size_t>(i) * n + k] = factor; // store L

                // Update within panel: columns k+1..jend-1
                const double* row_k = m + static_cast<size_t>(k) * n;
                double* row_i = m + static_cast<size_t>(i) * n;
                for (int j = k + 1; j < jend; j++) {
                    row_i[j] -= factor * row_k[j];
                }
            }
        }

        // ---- Step 2: DTRSM — solve L11 * U12 = A12 for panel rows ----
        // For each trailing column c >= jend, forward-substitute through
        // the panel's L11 to get the correct U12 values.
        // L11 is unit lower triangular in columns jb..jend-1.
        if (jend < n) {
            for (int j = jb; j < jend; j++) {
                double* row_j = m + static_cast<size_t>(j) * n;
                double factor_cache = 0.0;
                (void)factor_cache;
                for (int i = j + 1; i < jend; i++) {
                    double L_ij = m[static_cast<size_t>(i) * n + j];
                    double* row_i = m + static_cast<size_t>(i) * n;
                    // Update trailing columns of row i
                    for (int c = jend; c < n; c++) {
                        row_i[c] -= L_ij * row_j[c];
                    }
                }
            }

            // ---- Step 3: DGEMM — A22 -= L21 * U12 ----
            // L21: rows jend..n-1, columns jb..jend-1 (already computed)
            // U12: rows jb..jend-1, columns jend..n-1 (computed in step 2)
            // A22: rows jend..n-1, columns jend..n-1
            //
            // This is the BLAS-3 heart: matrix-matrix multiply.
            // Inner dimension = nb_actual, reusing each U12 row nb_actual times.
            int nb_actual = jend - jb;

            // Loop order: i, j, c (row-major optimal).
            // For each row i, load L21[i][j] once (scalar), then sweep
            // the entire U12 row j sequentially — cache-line friendly.
            for (int i = jend; i < n; i++) {
                double* row_i = m + static_cast<size_t>(i) * n;
                for (int j = jb; j < jend; j++) {
                    double L_ij = m[static_cast<size_t>(i) * n + j];
                    const double* U_row_j = m + static_cast<size_t>(j) * n;
                    for (int c = jend; c < n; c++) {
                        row_i[c] -= L_ij * U_row_j[c];
                    }
                }
            }
        }
    }
    return true;
}

// --------------------------------------------------------------------------
// Dense QR Decomposition (Householder reflections)
// --------------------------------------------------------------------------

DenseQR dense_qr_decompose(const DenseMatrix& a) {
    const int n = a.n;

    DenseQR qr;
    qr.n = n;
    qr.r_data = a.data; // copy A into R, will be transformed in-place
    qr.qt_data.assign(static_cast<size_t>(n) * n, 0.0);

    // Initialize Q^T = I
    for (int i = 0; i < n; i++)
        qr.qt_data[static_cast<size_t>(i) * n + i] = 1.0;

    double* R = qr.r_data.data();
    double* QT = qr.qt_data.data();

    std::vector<double> v(n); // Householder vector

    for (int k = 0; k < n; k++) {
        // Compute Householder vector for column k
        double sigma = 0.0;
        for (int i = k; i < n; i++) {
            double val = R[static_cast<size_t>(i) * n + k];
            sigma += val * val;
        }
        sigma = std::sqrt(sigma);

        if (sigma == 0.0) continue;

        double r_kk = R[static_cast<size_t>(k) * n + k];
        if (r_kk > 0.0) sigma = -sigma; // sign choice for stability

        // v = column k below diagonal, with v[k] modified
        for (int i = 0; i < k; i++) v[i] = 0.0;
        for (int i = k; i < n; i++) v[i] = R[static_cast<size_t>(i) * n + k];
        v[k] -= sigma;

        // Normalize: beta = 2 / (v^T v)
        double vTv = 0.0;
        for (int i = k; i < n; i++) vTv += v[i] * v[i];
        if (vTv == 0.0) continue;
        double beta = 2.0 / vTv;

        // Apply H = I - beta * v * v^T to R from left: R <- H * R
        for (int j = k; j < n; j++) {
            double dot = 0.0;
            for (int i = k; i < n; i++) dot += v[i] * R[static_cast<size_t>(i) * n + j];
            dot *= beta;
            for (int i = k; i < n; i++) R[static_cast<size_t>(i) * n + j] -= dot * v[i];
        }

        // Apply H to Q^T from left: Q^T <- H * Q^T
        for (int j = 0; j < n; j++) {
            double dot = 0.0;
            for (int i = k; i < n; i++) dot += v[i] * QT[static_cast<size_t>(i) * n + j];
            dot *= beta;
            for (int i = k; i < n; i++) QT[static_cast<size_t>(i) * n + j] -= dot * v[i];
        }
    }

    return qr;
}

void dense_qr_solve(const DenseQR& qr, double* b) {
    const int n = qr.n;
    const double* QT = qr.qt_data.data();
    const double* R = qr.r_data.data();

    // Compute Q^T * b
    std::vector<double> qtb(n, 0.0);
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += QT[static_cast<size_t>(i) * n + j] * b[j];
        }
        qtb[i] = sum;
    }

    // Back substitution: R * x = Q^T * b
    for (int i = n - 1; i >= 0; i--) {
        double sum = qtb[i];
        for (int j = i + 1; j < n; j++) {
            sum -= R[static_cast<size_t>(i) * n + j] * qtb[j];
        }
        double diag = R[static_cast<size_t>(i) * n + i];
        if (diag == 0.0) throw std::runtime_error("QR: singular R");
        qtb[i] = sum / diag;
    }

    for (int i = 0; i < n; i++) b[i] = qtb[i];
}

std::vector<double> dense_qr_solve(const DenseQR& qr, const std::vector<double>& b) {
    if (static_cast<int>(b.size()) != qr.n)
        throw std::invalid_argument("RHS size mismatch");
    std::vector<double> x = b;
    dense_qr_solve(qr, x.data());
    return x;
}

} // namespace dense
