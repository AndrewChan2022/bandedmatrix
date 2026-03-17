#include "../src/banded_matrix.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    Timer() : t0(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// Build a diagonally dominant banded matrix
static void fill_matrix_point(banded::BandMatrix& A, int n, int m1, int m2) {
    std::srand(42);
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
}

static void fill_matrix_blocked(banded::BandMatrixBlocked& A, int n, int m1, int m2) {
    std::srand(42); // same seed for same matrix
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
}

// Compute residual using original matrix values (regenerated with same seed)
static double compute_residual(int n, int m1, int m2,
                                const std::vector<double>& x,
                                const std::vector<double>& b) {
    std::srand(42);
    double max_res = 0.0;
    for (int i = 0; i < n; i++) {
        double row_sum_abs = 0.0;
        int j_lo = std::max(0, i - m1);
        int j_hi = std::min(n - 1, i + m2);
        // Regenerate same values
        double row_val = 0.0;
        for (int j = j_lo; j <= j_hi; j++) {
            if (j != i) {
                double val = (std::rand() % 200 - 100) / 100.0;
                row_val += val * x[j];
                row_sum_abs += std::abs(val);
            }
        }
        double diag = row_sum_abs + 1.0;
        row_val += diag * x[i];
        max_res = std::max(max_res, std::abs(row_val - b[i]));
    }
    return max_res;
}

int main() {
    std::printf("=== Block Algorithm Benchmark: Point vs Blocked LU ===\n\n");
    std::printf("History: LAPACK's milestone was reorganizing LU decomposition\n");
    std::printf("from column-by-column (BLAS-2) to panel + trailing update (BLAS-3).\n");
    std::printf("The DGEMM-like trailing update keeps data in cache, giving\n");
    std::printf("significant speedup at large bandwidths.\n\n");

    const int n = 100000;

    std::printf("N = %d. Varying bandwidth and block size.\n\n", n);

    // --- Part 1: Point vs Blocked across bandwidths ---
    std::printf("=== Part 1: Point vs Blocked (NB=32) ===\n\n");
    std::printf("  %6s  %12s  %12s  %12s  %12s  %10s\n",
                "Band", "Point(ms)", "Blocked(ms)", "Speedup", "Residual(P)", "Residual(B)");
    std::printf("  %6s  %12s  %12s  %12s  %12s  %10s\n",
                "----", "---------", "----------", "-------", "-----------", "-----------");

    for (int band : {1, 2, 3, 4, 8, 16, 32, 64}) {
        int m1 = band, m2 = band;

        std::vector<double> b(n);
        std::srand(12345);
        for (int i = 0; i < n; i++) b[i] = (std::rand() % 200 - 100) / 10.0;

        // Point algorithm
        double point_ms, point_res;
        {
            banded::BandMatrix A(n, m1, m2);
            fill_matrix_point(A, n, m1, m2);
            Timer t;
            banded::band_lu_decompose(A);
            auto x = banded::band_lu_solve(A, b);
            point_ms = t.elapsed_ms();
            point_res = compute_residual(n, m1, m2, x, b);
        }

        // Blocked algorithm (NB=32)
        double blocked_ms, blocked_res;
        {
            banded::BandMatrixBlocked A(n, m1, m2);
            fill_matrix_blocked(A, n, m1, m2);
            Timer t;
            banded::band_lu_decompose_blocked(A, 32);
            auto x = banded::band_lu_solve_blocked(A, b);
            blocked_ms = t.elapsed_ms();
            blocked_res = compute_residual(n, m1, m2, x, b);
        }

        double speedup = point_ms / std::max(blocked_ms, 1e-9);
        std::printf("  %6d  %12.2f  %12.2f  %11.2fx  %12.2e  %10.2e\n",
                    band, point_ms, blocked_ms, speedup, point_res, blocked_res);
    }

    // --- Part 2: Varying block size at band=32 ---
    std::printf("\n=== Part 2: Varying Block Size (band=32) ===\n\n");
    {
        int band = 32, m1 = band, m2 = band;

        std::vector<double> b(n);
        std::srand(12345);
        for (int i = 0; i < n; i++) b[i] = (std::rand() % 200 - 100) / 10.0;

        std::printf("  %6s  %12s  %12s\n", "NB", "Total(ms)", "Residual");
        std::printf("  %6s  %12s  %12s\n", "----", "---------", "--------");

        // Point (NB=1 equivalent)
        {
            banded::BandMatrix A(n, m1, m2);
            fill_matrix_point(A, n, m1, m2);
            Timer t;
            banded::band_lu_decompose(A);
            auto x = banded::band_lu_solve(A, b);
            double ms = t.elapsed_ms();
            double res = compute_residual(n, m1, m2, x, b);
            std::printf("  %6s  %12.2f  %12.2e\n", "point", ms, res);
        }

        for (int nb : {4, 8, 16, 32, 64}) {
            banded::BandMatrixBlocked A(n, m1, m2);
            fill_matrix_blocked(A, n, m1, m2);
            Timer t;
            banded::band_lu_decompose_blocked(A, nb);
            auto x = banded::band_lu_solve_blocked(A, b);
            double ms = t.elapsed_ms();
            double res = compute_residual(n, m1, m2, x, b);
            std::printf("  %6d  %12.2f  %12.2e\n", nb, ms, res);
        }
    }

    // --- Part 3: Varying block size at band=64 ---
    std::printf("\n=== Part 3: Varying Block Size (band=64) ===\n\n");
    {
        int band = 64, m1 = band, m2 = band;

        std::vector<double> b(n);
        std::srand(12345);
        for (int i = 0; i < n; i++) b[i] = (std::rand() % 200 - 100) / 10.0;

        std::printf("  %6s  %12s  %12s\n", "NB", "Total(ms)", "Residual");
        std::printf("  %6s  %12s  %12s\n", "----", "---------", "--------");

        {
            banded::BandMatrix A(n, m1, m2);
            fill_matrix_point(A, n, m1, m2);
            Timer t;
            banded::band_lu_decompose(A);
            auto x = banded::band_lu_solve(A, b);
            double ms = t.elapsed_ms();
            double res = compute_residual(n, m1, m2, x, b);
            std::printf("  %6s  %12.2f  %12.2e\n", "point", ms, res);
        }

        for (int nb : {4, 8, 16, 32, 64}) {
            banded::BandMatrixBlocked A(n, m1, m2);
            fill_matrix_blocked(A, n, m1, m2);
            Timer t;
            banded::band_lu_decompose_blocked(A, nb);
            auto x = banded::band_lu_solve_blocked(A, b);
            double ms = t.elapsed_ms();
            double res = compute_residual(n, m1, m2, x, b);
            std::printf("  %6d  %12.2f  %12.2e\n", nb, ms, res);
        }
    }

    std::printf("\nDone.\n");
    return 0;
}
