#include "../src/dense_matrix.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>

struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    Timer() : t0(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

static void fill_matrix(dense::DenseMatrix& A, int n) {
    std::srand(42);
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++) {
            if (j != i) {
                double val = (std::rand() % 200 - 100) / (100.0 * n);
                A(i, j) = val;
                row_sum += std::abs(val);
            }
        }
        A(i, i) = row_sum + 1.0;
    }
}

static double compute_residual(int n, const dense::DenseMatrix& A_orig,
                                const std::vector<double>& x,
                                const std::vector<double>& b) {
    double max_res = 0.0;
    for (int i = 0; i < n; i++) {
        double row_val = 0.0;
        for (int j = 0; j < n; j++) {
            row_val += A_orig(i, j) * x[j];
        }
        max_res = std::max(max_res, std::abs(row_val - b[i]));
    }
    return max_res;
}

int main() {
    std::printf("=== Dense Matrix: Point vs Blocked LU Benchmark ===\n\n");
    std::printf("LAPACK's block algorithm reorganizes LU decomposition:\n");
    std::printf("  Point: column-by-column elimination (BLAS-2, memory-bound)\n");
    std::printf("  Blocked: panel factorization + DGEMM trailing update (BLAS-3, cache-optimal)\n\n");

    // --- Part 1: Varying N with fixed NB ---
    std::printf("=== Part 1: Point vs Blocked (NB=64) ===\n\n");
    std::printf("  %6s  %12s  %12s  %10s  %12s  %12s\n",
                "N", "Point(ms)", "Blocked(ms)", "Speedup", "Res(Point)", "Res(Block)");
    std::printf("  %6s  %12s  %12s  %10s  %12s  %12s\n",
                "---", "---------", "----------", "-------", "----------", "----------");

    for (int n : {100, 200, 500, 1000, 2000}) {
        std::vector<double> b(n);
        std::srand(12345);
        for (int i = 0; i < n; i++) b[i] = (std::rand() % 200 - 100) / 10.0;

        // Save original matrix for residual computation
        dense::DenseMatrix A_orig(n);
        fill_matrix(A_orig, n);

        // Point algorithm
        double point_ms, point_res;
        {
            dense::DenseMatrix A(n);
            fill_matrix(A, n);
            Timer t;
            dense::dense_lu_decompose(A);
            auto x = dense::dense_lu_solve(A, b);
            point_ms = t.elapsed_ms();
            point_res = compute_residual(n, A_orig, x, b);
        }

        // Blocked algorithm
        double blocked_ms, blocked_res;
        {
            dense::DenseMatrix A(n);
            fill_matrix(A, n);
            Timer t;
            dense::dense_lu_decompose_blocked(A, 64);
            auto x = dense::dense_lu_solve(A, b);
            blocked_ms = t.elapsed_ms();
            blocked_res = compute_residual(n, A_orig, x, b);
        }

        double speedup = point_ms / std::max(blocked_ms, 1e-9);
        std::printf("  %6d  %12.2f  %12.2f  %9.2fx  %12.2e  %12.2e\n",
                    n, point_ms, blocked_ms, speedup, point_res, blocked_res);
    }

    // --- Part 2: Varying NB at N=1000 ---
    std::printf("\n=== Part 2: Varying Block Size (N=1000) ===\n\n");
    {
        const int n = 1000;
        std::vector<double> b(n);
        std::srand(12345);
        for (int i = 0; i < n; i++) b[i] = (std::rand() % 200 - 100) / 10.0;

        dense::DenseMatrix A_orig(n);
        fill_matrix(A_orig, n);

        std::printf("  %6s  %12s  %12s\n", "NB", "Total(ms)", "Residual");
        std::printf("  %6s  %12s  %12s\n", "----", "---------", "--------");

        // Point
        {
            dense::DenseMatrix A(n);
            fill_matrix(A, n);
            Timer t;
            dense::dense_lu_decompose(A);
            auto x = dense::dense_lu_solve(A, b);
            std::printf("  %6s  %12.2f  %12.2e\n", "point",
                        t.elapsed_ms(), compute_residual(n, A_orig, x, b));
        }

        for (int nb : {8, 16, 32, 64, 128, 256}) {
            dense::DenseMatrix A(n);
            fill_matrix(A, n);
            Timer t;
            dense::dense_lu_decompose_blocked(A, nb);
            auto x = dense::dense_lu_solve(A, b);
            std::printf("  %6d  %12.2f  %12.2e\n", nb,
                        t.elapsed_ms(), compute_residual(n, A_orig, x, b));
        }
    }

    // --- Part 3: Varying NB at N=2000 ---
    std::printf("\n=== Part 3: Varying Block Size (N=2000) ===\n\n");
    {
        const int n = 2000;
        std::vector<double> b(n);
        std::srand(12345);
        for (int i = 0; i < n; i++) b[i] = (std::rand() % 200 - 100) / 10.0;

        dense::DenseMatrix A_orig(n);
        fill_matrix(A_orig, n);

        std::printf("  %6s  %12s  %12s\n", "NB", "Total(ms)", "Residual");
        std::printf("  %6s  %12s  %12s\n", "----", "---------", "--------");

        {
            dense::DenseMatrix A(n);
            fill_matrix(A, n);
            Timer t;
            dense::dense_lu_decompose(A);
            auto x = dense::dense_lu_solve(A, b);
            std::printf("  %6s  %12.2f  %12.2e\n", "point",
                        t.elapsed_ms(), compute_residual(n, A_orig, x, b));
        }

        for (int nb : {16, 32, 64, 128, 256}) {
            dense::DenseMatrix A(n);
            fill_matrix(A, n);
            Timer t;
            dense::dense_lu_decompose_blocked(A, nb);
            auto x = dense::dense_lu_solve(A, b);
            std::printf("  %6d  %12.2f  %12.2e\n", nb,
                        t.elapsed_ms(), compute_residual(n, A_orig, x, b));
        }
    }

    std::printf("\nDone.\n");
    return 0;
}
