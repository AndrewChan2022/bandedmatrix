#include "../src/banded_matrix.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>

struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    Timer() : t0(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

int main() {
    std::printf("=== Bandwidth Scaling Benchmark ===\n");
    std::printf("N = 100,000 for all tests. Varying bandwidth.\n\n");

    std::printf("  %6s  %12s  %12s  %12s  %14s  %10s\n",
                "Band", "Decomp(ms)", "Solve(ms)", "Total(ms)", "Residual", "Bytes/row");
    std::printf("  %6s  %12s  %12s  %12s  %14s  %10s\n",
                "----", "----------", "---------", "---------", "--------", "---------");

    const int n = 100000;

    for (int band : {1, 2, 3, 4, 8, 16, 32, 64}) {
        int m1 = band, m2 = band;
        int mm = m1 + m2 + 1;

        banded::BandMatrix A(n, m1, m2);
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
        std::vector<double> au_copy = A.au;

        std::vector<double> b(n);
        for (int i = 0; i < n; i++) b[i] = (std::rand() % 200 - 100) / 10.0;

        Timer t_dec;
        banded::band_lu_decompose(A);
        double dec_ms = t_dec.elapsed_ms();

        Timer t_sol;
        auto x = banded::band_lu_solve(A, b);
        double sol_ms = t_sol.elapsed_ms();

        // Residual
        double max_res = 0.0;
        for (int i = 0; i < n; i++) {
            double row_val = 0.0;
            int j_lo = std::max(0, i - m1);
            int j_hi = std::min(n - 1, i + m2);
            for (int j = j_lo; j <= j_hi; j++)
                row_val += au_copy[static_cast<size_t>(i) * mm + (j - i + m1)] * x[j];
            max_res = std::max(max_res, std::abs(row_val - b[i]));
        }

        int bytes_per_row = mm * 8; // doubles
        std::printf("  %6d  %12.2f  %12.2f  %12.2f  %14.2e  %10d\n",
                    band, dec_ms, sol_ms, dec_ms + sol_ms, max_res, bytes_per_row);
    }

    std::printf("\nNote: SIMD (AVX-256/512) becomes beneficial when bandwidth > ~16\n");
    std::printf("      (inner loops > 32 elements = 4+ AVX-512 vector widths).\n");
    std::printf("      At band=3 (7 elements), loops are too short for SIMD gain.\n");

    return 0;
}
