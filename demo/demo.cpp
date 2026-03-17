#include "../src/banded_matrix.h"
#include <cstdio>
#include <chrono>
#include <vector>

// Demonstrate banded matrix LU and QR solvers on a heat equation discretization.
// The 1D Laplacian on n interior points gives a tridiagonal system:
//   2 -1  0 ...
//  -1  2 -1 ...
//   0 -1  2 ...
//   ...       2

int main() {
    std::printf("=== Banded Matrix Solver Demo ===\n\n");

    // --- Small example: 5x5 tridiagonal ---
    {
        const int n = 5;
        std::printf("1. Small tridiagonal system (n=%d, m1=1, m2=1)\n", n);
        std::printf("   Matrix: 1D Laplacian (heat equation)\n\n");

        // Build matrix
        banded::BandMatrix A(n, 1, 1);
        for (int i = 0; i < n; i++) {
            A(i, i) = 2.0;
            if (i > 0) A(i, i - 1) = -1.0;
            if (i < n - 1) A(i, i + 1) = -1.0;
        }

        std::vector<double> b = {1.0, 0.0, 0.0, 0.0, 1.0};
        std::printf("   RHS b = [");
        for (int i = 0; i < n; i++) std::printf(" %.1f", b[i]);
        std::printf(" ]\n");

        // LU solve
        banded::BandMatrix A_lu = A; // copy for LU (modifies in place)
        banded::band_lu_decompose(A_lu);
        auto x_lu = banded::band_lu_solve(A_lu, b);

        std::printf("   LU solution x = [");
        for (int i = 0; i < n; i++) std::printf(" %.6f", x_lu[i]);
        std::printf(" ]\n");

        // QR solve
        auto qr = banded::band_qr_decompose(A);
        auto x_qr = banded::band_qr_solve(qr, b);

        std::printf("   QR solution x = [");
        for (int i = 0; i < n; i++) std::printf(" %.6f", x_qr[i]);
        std::printf(" ]\n\n");
    }

    // --- Performance benchmark ---
    {
        std::printf("2. Performance benchmark\n\n");

        struct BenchCase { int n; int m1; int m2; const char* label; };
        BenchCase cases[] = {
            {10000,   1, 1, "tridiag   n=10K  "},
            {100000,  1, 1, "tridiag   n=100K "},
            {1000000, 1, 1, "tridiag   n=1M   "},
            {10000,   3, 3, "7-diag    n=10K  "},
            {100000,  3, 3, "7-diag    n=100K "},
            {10000,   5, 5, "11-diag   n=10K  "},
            {100000,  5, 5, "11-diag   n=100K "},
        };

        std::printf("   %-25s  %10s  %12s\n", "System", "LU (ms)", "Throughput");
        std::printf("   %-25s  %10s  %12s\n", "------", "-------", "----------");

        for (auto& c : cases) {
            banded::BandMatrix A(c.n, c.m1, c.m2);

            // Diagonally dominant band matrix
            for (int i = 0; i < c.n; i++) {
                double diag = 0.0;
                for (int d = 1; d <= c.m1; d++) {
                    if (i - d >= 0) { A(i, i - d) = -1.0 / d; diag += 1.0 / d; }
                }
                for (int d = 1; d <= c.m2; d++) {
                    if (i + d < c.n) { A(i, i + d) = -1.0 / d; diag += 1.0 / d; }
                }
                A(i, i) = diag + 1.0;
            }

            std::vector<double> b(c.n, 1.0);

            auto t0 = std::chrono::high_resolution_clock::now();
            banded::band_lu_decompose(A);
            auto x = banded::band_lu_solve(A, b);
            auto t1 = std::chrono::high_resolution_clock::now();

            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double mrows_per_sec = c.n / (ms * 1000.0); // millions of rows / sec

            std::printf("   %-25s  %8.2f ms  %8.1f Mrows/s\n",
                        c.label, ms, mrows_per_sec);
        }
    }

    std::printf("\nDone.\n");
    return 0;
}
