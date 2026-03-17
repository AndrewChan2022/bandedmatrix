#include "banded_matrix.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace banded {

// --------------------------------------------------------------------------
// LU Decomposition (Numerical Recipes Ch 2.4, bandec)
// --------------------------------------------------------------------------
// Input compact storage: au[i * mm + (j - i + m1)] = A[i][j]
// where mm = m1 + m2 + 1, diagonal at column m1.
//
// After decomposition:
//   au is rearranged so diagonal is at column 0.
//   al[i * m1 + k] stores L multipliers.
//   pivot[k] stores pivot row index.

bool band_lu_decompose(BandMatrix& a) {
    const int n = a.n;
    const int m1 = a.m1;
    const int m2 = a.m2;
    const int mm = a.mm; // m1 + m2 + 1
    double* au = a.au.data();
    double* al = a.al.data();

    a.d = 1.0;

    // Rearrange the first m1 rows: shift band elements left so that
    // the diagonal ends up at column 0 after the rearrangement.
    // Row i (i < m1) has only mm - (m1 - i) valid elements starting
    // at column (m1 - i). Shift them left and zero-fill the right.
    for (int i = 0; i < m1; i++) {
        double* row = au + static_cast<size_t>(i) * mm;
        int shift = m1 - i;
        for (int j = shift; j < mm; j++) {
            row[j - shift] = row[j];
        }
        for (int j = mm - shift; j < mm; j++) {
            row[j] = 0.0;
        }
    }

    // Main elimination loop
    for (int k = 0; k < n; k++) {
        double* row_k = au + static_cast<size_t>(k) * mm;

        // Find pivot in column k: search rows k..min(k+m1, n-1).
        // After rearrangement, the "column k" element for row r
        // is at au[r * mm + 0] (because each row's leftmost is
        // its current pivot candidate).
        double pivot_val = row_k[0];
        int pivot_row = k;
        int search_depth = std::min(m1, n - 1 - k);

        for (int i = k + 1; i <= k + search_depth; i++) {
            double* row_i = au + static_cast<size_t>(i) * mm;
            if (std::abs(row_i[0]) > std::abs(pivot_val)) {
                pivot_val = row_i[0];
                pivot_row = i;
            }
        }

        a.pivot[k] = pivot_row;

        if (pivot_val == 0.0) {
            return false; // singular
        }

        // Swap rows k and pivot_row if needed
        if (pivot_row != k) {
            a.d = -a.d;
            double* row_p = au + static_cast<size_t>(pivot_row) * mm;
            for (int j = 0; j < mm; j++) {
                std::swap(row_k[j], row_p[j]);
            }
        }

        // Precompute reciprocal of pivot (multiply is ~4x faster than divide)
        double inv_pivot = 1.0 / row_k[0];
        a.diag_inv[k] = inv_pivot;

        // Eliminate: for each row below pivot within bandwidth
        for (int i = 1; i <= search_depth; i++) {
            double* row_ki = au + static_cast<size_t>(k + i) * mm;
            double factor = row_ki[0] * inv_pivot;

            // Store L factor
            al[static_cast<size_t>(k) * m1 + (i - 1)] = factor;

            // Update row and shift left by 1
            for (int j = 1; j < mm; j++) {
                row_ki[j - 1] = row_ki[j] - factor * row_k[j];
            }
            row_ki[mm - 1] = 0.0;
        }
    }

    return true;
}

// --------------------------------------------------------------------------
// LU Solve (Numerical Recipes Ch 2.4, banbks)
// --------------------------------------------------------------------------

void band_lu_solve(const BandMatrix& a, double* b) {
    const int n = a.n;
    const int m1 = a.m1;
    const int m2 = a.m2;
    const int mm = a.mm;
    const double* au = a.au.data();
    const double* al = a.al.data();

    // Forward substitution (L * y = Pb)
    for (int k = 0; k < n; k++) {
        int p = a.pivot[k];
        if (p != k) {
            std::swap(b[k], b[p]);
        }

        int depth = std::min(m1, n - 1 - k);
        for (int i = 1; i <= depth; i++) {
            b[k + i] -= al[static_cast<size_t>(k) * m1 + (i - 1)] * b[k];
        }
    }

    // Back substitution (U * x = y)
    // After rearrangement, au[k * mm + 0] is diagonal,
    // au[k * mm + j] for j=1..min(mm-1, n-1-k) are super-diagonal entries.
    // Uses precomputed diag_inv[] to replace division with multiplication.
    const double* dinv = a.diag_inv.data();
    for (int k = n - 1; k >= 0; k--) {
        const double* row_k = au + static_cast<size_t>(k) * mm;
        double sum = b[k];
        int width = std::min(mm - 1, n - 1 - k);
        for (int j = 1; j <= width; j++) {
            sum -= row_k[j] * b[k + j];
        }
        b[k] = sum * dinv[k];
    }
}

std::vector<double> band_lu_solve(const BandMatrix& a, const std::vector<double>& b) {
    if (static_cast<int>(b.size()) != a.n) {
        throw std::invalid_argument("RHS size does not match matrix dimension");
    }
    std::vector<double> x = b;
    band_lu_solve(a, x.data());
    return x;
}

// --------------------------------------------------------------------------
// QR Decomposition using Givens Rotations
// --------------------------------------------------------------------------
// For a band matrix with m1 sub-diags and m2 super-diags, we zero out
// sub-diagonal entries using Givens rotations. Each rotation in column k
// eliminates one sub-diagonal entry.
//
// R has upper bandwidth m1 + m2 (fill-in from rotations).
// We store R in compact band form with stride = m1 + m2 + 1.

BandQR band_qr_decompose(const BandMatrix& a) {
    const int n = a.n;
    const int m1 = a.m1;
    const int m2 = a.m2;
    const int bw = m1 + m2; // upper bandwidth of R

    BandQR qr;
    qr.n = n;
    qr.m1 = m1;
    qr.m2 = m2;
    qr.stride = bw + 1;

    // Working storage: need both lower and upper parts during elimination.
    // Use stride = 2*m1 + m2 + 1. Element at (i, j) maps to
    // work[i * w_stride + (j - i + m1)].
    const int w_stride = 2 * m1 + m2 + 1;
    std::vector<double> work(static_cast<size_t>(n) * w_stride, 0.0);

    // Copy A into work storage
    for (int i = 0; i < n; i++) {
        int j_lo = std::max(0, i - m1);
        int j_hi = std::min(n - 1, i + m2);
        for (int j = j_lo; j <= j_hi; j++) {
            work[static_cast<size_t>(i) * w_stride + (j - i + m1)] = a(i, j);
        }
    }

    // Reserve space for Givens rotations
    qr.cs.reserve(static_cast<size_t>(n) * m1);
    qr.sn.reserve(static_cast<size_t>(n) * m1);
    qr.n_rotations = 0;

    // Apply Givens rotations column by column
    for (int k = 0; k < n; k++) {
        int depth = std::min(m1, n - 1 - k);
        for (int i = 1; i <= depth; i++) {
            double* row_pivot = work.data() + static_cast<size_t>(k) * w_stride;
            double* row_elim = work.data() + static_cast<size_t>(k + i) * w_stride;

            double a_kk = row_pivot[m1];       // A[k][k] in work
            double a_ik = row_elim[m1 - i];    // A[k+i][k] in work

            if (a_ik == 0.0) {
                qr.cs.push_back(1.0);
                qr.sn.push_back(0.0);
                qr.n_rotations++;
                continue;
            }

            // Compute Givens rotation
            double r, c, s;
            if (std::abs(a_kk) >= std::abs(a_ik)) {
                double t = a_ik / a_kk;
                r = std::sqrt(1.0 + t * t);
                c = 1.0 / r;
                s = t * c;
            } else {
                double t = a_kk / a_ik;
                r = std::sqrt(1.0 + t * t);
                s = 1.0 / r;
                c = t * s;
            }

            qr.cs.push_back(c);
            qr.sn.push_back(s);
            qr.n_rotations++;

            // Apply rotation to rows k and k+i.
            // Bounds proof (no branching needed):
            //   idx_k = j-k+m1:   j in [k,k+bw] => idx_k in [m1, 2*m1+m2] = [m1, w_stride-1]
            //   idx_i = j-k-i+m1: j in [k,k+bw], i in [1,m1] => idx_i in [0, w_stride-1-i]
            //   Both always in [0, w_stride), so all bounds checks eliminated.
            int j_max = std::min(n - 1, k + bw);
            for (int j = k; j <= j_max; j++) {
                int idx_k = j - k + m1;
                int idx_i = j - (k + i) + m1;

                double val_k = row_pivot[idx_k];
                double val_i = row_elim[idx_i];

                row_pivot[idx_k] =  c * val_k + s * val_i;
                row_elim[idx_i]  = -s * val_k + c * val_i;
            }
        }
    }

    // Extract R into compact form: R[i][j] for j in [i, min(n-1, i+bw)]
    // stored at r_data[i * stride + (j - i)]
    qr.r_data.assign(static_cast<size_t>(n) * qr.stride, 0.0);
    for (int i = 0; i < n; i++) {
        const double* w_row = work.data() + static_cast<size_t>(i) * w_stride;
        double* r_row = qr.r_data.data() + static_cast<size_t>(i) * qr.stride;
        int j_hi = std::min(n - 1, i + bw);
        for (int j = i; j <= j_hi; j++) {
            int w_idx = j - i + m1;
            int r_idx = j - i;
            r_row[r_idx] = w_row[w_idx];
        }
    }

    return qr;
}

// --------------------------------------------------------------------------
// QR Solve: x = R^{-1} * Q^T * b
// --------------------------------------------------------------------------

void band_qr_solve(const BandQR& qr, double* b) {
    const int n = qr.n;
    const int m1 = qr.m1;
    const int bw = qr.m1 + qr.m2;

    // Apply Q^T to b: replay Givens rotations
    int rot_idx = 0;
    for (int k = 0; k < n; k++) {
        int depth = std::min(m1, n - 1 - k);
        for (int i = 1; i <= depth; i++) {
            double c = qr.cs[rot_idx];
            double s = qr.sn[rot_idx];
            rot_idx++;

            double bk = b[k];
            double bi = b[k + i];
            b[k]     =  c * bk + s * bi;
            b[k + i] = -s * bk + c * bi;
        }
    }

    // Back substitution with R.
    // Precompute reciprocal diagonals to replace division with multiplication.
    const double* r_base = qr.r_data.data();
    const int r_stride = qr.stride;
    for (int i = n - 1; i >= 0; i--) {
        const double* r_row = r_base + static_cast<size_t>(i) * r_stride;
        double sum = b[i];
        int j_hi = std::min(n - 1, i + bw);
        for (int j = i + 1; j <= j_hi; j++) {
            sum -= r_row[j - i] * b[j];
        }
        double diag = r_row[0];
        if (diag == 0.0) {
            throw std::runtime_error("QR solve: singular R matrix");
        }
        b[i] = sum / diag;
    }
}

std::vector<double> band_qr_solve(const BandQR& qr, const std::vector<double>& b) {
    if (static_cast<int>(b.size()) != qr.n) {
        throw std::invalid_argument("RHS size does not match matrix dimension");
    }
    std::vector<double> x = b;
    band_qr_solve(qr, x.data());
    return x;
}

} // namespace banded
