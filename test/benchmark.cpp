#include "../src/banded_matrix.h"
#include "../src/dense_matrix.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

static double now_ms() {
    static auto t0 = std::chrono::high_resolution_clock::now();
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t - t0).count();
}

struct Timer {
    double start;
    Timer() : start(now_ms()) {}
    double elapsed() const { return now_ms() - start; }
};

// Build a diagonally dominant banded system in both dense and band form.
// Returns max |A*x - b| residual norm.
static double compute_residual_band(int n, int m1, int m2,
                                     const std::vector<double>& x,
                                     const std::vector<double>& b,
                                     const std::vector<double>& a_flat,
                                     int mm) {
    double max_res = 0.0;
    for (int i = 0; i < n; i++) {
        double row_val = 0.0;
        int j_lo = std::max(0, i - m1);
        int j_hi = std::min(n - 1, i + m2);
        for (int j = j_lo; j <= j_hi; j++) {
            row_val += a_flat[static_cast<size_t>(i) * mm + (j - i + m1)] * x[j];
        }
        max_res = std::max(max_res, std::abs(row_val - b[i]));
    }
    return max_res;
}

static double compute_residual_dense(int n,
                                      const std::vector<double>& x,
                                      const std::vector<double>& b,
                                      const std::vector<double>& a_flat) {
    double max_res = 0.0;
    for (int i = 0; i < n; i++) {
        double row_val = 0.0;
        for (int j = 0; j < n; j++) {
            row_val += a_flat[static_cast<size_t>(i) * n + j] * x[j];
        }
        max_res = std::max(max_res, std::abs(row_val - b[i]));
    }
    return max_res;
}

// --------------------------------------------------------------------------
// Benchmark functions
// --------------------------------------------------------------------------

struct BenchResult {
    const char* method;
    int n;
    int bandwidth;
    double decompose_ms;
    double solve_ms;
    double total_ms;
    double residual;
    bool ok;
};

static void print_header() {
    std::printf("%-20s %10s %6s %12s %12s %12s %12s\n",
                "Method", "N", "Band", "Decomp(ms)", "Solve(ms)", "Total(ms)", "Residual");
    std::printf("%-20s %10s %6s %12s %12s %12s %12s\n",
                "------", "---", "----", "----------", "---------", "---------", "--------");
}

static void print_result(const BenchResult& r) {
    std::printf("%-20s %10d %6d %12.3f %12.3f %12.3f %12.2e %s\n",
                r.method, r.n, r.bandwidth,
                r.decompose_ms, r.solve_ms, r.total_ms,
                r.residual, r.ok ? "OK" : "FAIL");
}

// Banded LU benchmark
static BenchResult bench_banded_lu(int n, int band) {
    int m1 = band, m2 = band;
    int mm = m1 + m2 + 1;

    // Build matrix
    banded::BandMatrix A(n, m1, m2);
    std::vector<double> a_copy(A.au); // save for residual before copy is valid

    std::srand(12345);
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        int j_lo = std::max(0, i - m1);
        int j_hi = std::min(n - 1, i + m2);
        for (int j = j_lo; j <= j_hi; j++) {
            if (j != i) {
                double val = (std::rand() % 200 - 100) / 100.0;
                A(i, j) = val;
                row_sum += std::abs(val);
            }
        }
        A(i, i) = row_sum + 1.0;
    }
    a_copy = A.au; // save after filling

    std::vector<double> b(n);
    for (int i = 0; i < n; i++) b[i] = (std::rand() % 200 - 100) / 10.0;

    BenchResult res;
    res.method = "Banded LU";
    res.n = n;
    res.bandwidth = band;

    Timer t_dec;
    res.ok = banded::band_lu_decompose(A);
    res.decompose_ms = t_dec.elapsed();

    Timer t_sol;
    auto x = banded::band_lu_solve(A, b);
    res.solve_ms = t_sol.elapsed();
    res.total_ms = res.decompose_ms + res.solve_ms;

    // Compute residual using saved copy
    res.residual = compute_residual_band(n, m1, m2, x, b, a_copy, mm);

    return res;
}

// Banded QR benchmark
static BenchResult bench_banded_qr(int n, int band) {
    int m1 = band, m2 = band;
    int mm = m1 + m2 + 1;

    banded::BandMatrix A(n, m1, m2);
    std::srand(12345); // same seed as LU for same matrix
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        int j_lo = std::max(0, i - m1);
        int j_hi = std::min(n - 1, i + m2);
        for (int j = j_lo; j <= j_hi; j++) {
            if (j != i) {
                double val = (std::rand() % 200 - 100) / 100.0;
                A(i, j) = val;
                row_sum += std::abs(val);
            }
        }
        A(i, i) = row_sum + 1.0;
    }
    std::vector<double> a_copy = A.au;

    std::vector<double> b(n);
    for (int i = 0; i < n; i++) b[i] = (std::rand() % 200 - 100) / 10.0;

    BenchResult res;
    res.method = "Banded QR";
    res.n = n;
    res.bandwidth = band;
    res.ok = true;

    Timer t_dec;
    auto qr = banded::band_qr_decompose(A);
    res.decompose_ms = t_dec.elapsed();

    Timer t_sol;
    auto x = banded::band_qr_solve(qr, b);
    res.solve_ms = t_sol.elapsed();
    res.total_ms = res.decompose_ms + res.solve_ms;

    res.residual = compute_residual_band(n, m1, m2, x, b, a_copy, mm);

    return res;
}

// Dense LU benchmark
static BenchResult bench_dense_lu(int n, int band) {
    int m1 = band, m2 = band;

    dense::DenseMatrix A(n);
    std::srand(12345);
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        int j_lo = std::max(0, i - m1);
        int j_hi = std::min(n - 1, i + m2);
        for (int j = j_lo; j <= j_hi; j++) {
            if (j != i) {
                double val = (std::rand() % 200 - 100) / 100.0;
                A(i, j) = val;
                row_sum += std::abs(val);
            }
        }
        A(i, i) = row_sum + 1.0;
    }
    std::vector<double> a_copy = A.data;

    std::vector<double> b(n);
    for (int i = 0; i < n; i++) b[i] = (std::rand() % 200 - 100) / 10.0;

    BenchResult res;
    res.method = "Dense LU";
    res.n = n;
    res.bandwidth = band;

    Timer t_dec;
    res.ok = dense::dense_lu_decompose(A);
    res.decompose_ms = t_dec.elapsed();

    Timer t_sol;
    auto x = dense::dense_lu_solve(A, b);
    res.solve_ms = t_sol.elapsed();
    res.total_ms = res.decompose_ms + res.solve_ms;

    res.residual = compute_residual_dense(n, x, b, a_copy);

    return res;
}

// Dense QR benchmark
static BenchResult bench_dense_qr(int n, int band) {
    int m1 = band, m2 = band;

    dense::DenseMatrix A(n);
    std::srand(12345);
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        int j_lo = std::max(0, i - m1);
        int j_hi = std::min(n - 1, i + m2);
        for (int j = j_lo; j <= j_hi; j++) {
            if (j != i) {
                double val = (std::rand() % 200 - 100) / 100.0;
                A(i, j) = val;
                row_sum += std::abs(val);
            }
        }
        A(i, i) = row_sum + 1.0;
    }
    std::vector<double> a_copy = A.data;

    std::vector<double> b(n);
    for (int i = 0; i < n; i++) b[i] = (std::rand() % 200 - 100) / 10.0;

    BenchResult res;
    res.method = "Dense QR";
    res.n = n;
    res.bandwidth = band;
    res.ok = true;

    Timer t_dec;
    auto qr = dense::dense_qr_decompose(A);
    res.decompose_ms = t_dec.elapsed();

    Timer t_sol;
    auto x = dense::dense_qr_solve(qr, b);
    res.solve_ms = t_sol.elapsed();
    res.total_ms = res.decompose_ms + res.solve_ms;

    res.residual = compute_residual_dense(n, x, b, a_copy);

    return res;
}

int main() {
    std::printf("=== Banded vs Dense Matrix Solver Benchmark ===\n\n");
    std::printf("All matrices are diagonally dominant with the same random entries.\n");
    std::printf("Band = number of sub/super diagonals (bandwidth = 2*band+1).\n\n");

    const int band = 3; // m1 = m2 = 3, so bandwidth = 7

    // ---- a. Small matrix ----
    {
        const int n = 10;
        std::printf("--- (a) Small matrix: n = %d, band = %d ---\n\n", n, band);
        print_header();
        print_result(bench_dense_lu(n, band));
        print_result(bench_dense_qr(n, band));
        print_result(bench_banded_lu(n, band));
        print_result(bench_banded_qr(n, band));
        std::printf("\n");
    }

    // ---- b. Middle scale: 1000 x 1000 ----
    {
        const int n = 1000;
        std::printf("--- (b) Middle scale: n = %d, band = %d ---\n\n", n, band);
        print_header();
        print_result(bench_dense_lu(n, band));
        print_result(bench_dense_qr(n, band));
        print_result(bench_banded_lu(n, band));
        print_result(bench_banded_qr(n, band));
        std::printf("\n");
    }

    // ---- d. Large scale: 1,000,000 x 1,000,000 ----
    // Dense is infeasible at this size (would need ~7.5 TB RAM).
    // Only banded solvers run.
    {
        const int n = 1000000;
        std::printf("--- (d) Large scale: n = %d, band = %d ---\n\n", n, band);
        std::printf("NOTE: Dense solvers skipped (n^2 = %.0e doubles = %.1f TB RAM).\n\n",
                    (double)n * n, (double)n * n * 8.0 / 1e12);
        print_header();
        print_result(bench_banded_lu(n, band));
        print_result(bench_banded_qr(n, band));
        std::printf("\n");
    }

    // ---- Speedup summary ----
    {
        std::printf("=== Speedup Summary (Dense LU vs Banded LU) ===\n\n");
        for (int test_n : {10, 100, 1000}) {
            auto dense_r = bench_dense_lu(test_n, band);
            auto band_r = bench_banded_lu(test_n, band);
            double speedup = dense_r.total_ms / std::max(band_r.total_ms, 1e-6);
            std::printf("  n=%5d: Dense LU = %10.3f ms, Banded LU = %10.3f ms, Speedup = %.1fx\n",
                        test_n, dense_r.total_ms, band_r.total_ms, speedup);
        }
        std::printf("\n");
    }

    std::printf("Done.\n");
    return 0;
}
