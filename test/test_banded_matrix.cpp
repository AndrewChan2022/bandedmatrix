#include "../src/banded_matrix.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <chrono>

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define ASSERT_NEAR(a, b, tol)                                                \
    do {                                                                       \
        double _a = (a), _b = (b), _t = (tol);                                \
        if (std::abs(_a - _b) > _t) {                                         \
            std::printf("  FAIL: %s:%d: |%.12g - %.12g| = %.3e > %.3e\n",     \
                        __FILE__, __LINE__, _a, _b, std::abs(_a - _b), _t);   \
            g_tests_failed++;                                                  \
            return;                                                            \
        }                                                                      \
    } while (0)

#define ASSERT_TRUE(cond)                                                     \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("  FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            g_tests_failed++;                                                  \
            return;                                                            \
        }                                                                      \
    } while (0)

#define TEST(name)                                                            \
    static void test_##name();                                                 \
    struct Register_##name {                                                    \
        Register_##name() {                                                    \
            std::printf("Running %s...\n", #name);                             \
            test_##name();                                                     \
            g_tests_passed++;                                                  \
        }                                                                      \
    } g_reg_##name;                                                            \
    static void test_##name()

// --------------------------------------------------------------------------
// Helper: multiply dense band matrix A * x to verify solutions
// --------------------------------------------------------------------------
static std::vector<double> band_matvec(int n, int m1, int m2,
                                        const std::vector<std::vector<double>>& A,
                                        const std::vector<double>& x) {
    std::vector<double> y(n, 0.0);
    for (int i = 0; i < n; i++) {
        int j_lo = std::max(0, i - m1);
        int j_hi = std::min(n - 1, i + m2);
        for (int j = j_lo; j <= j_hi; j++) {
            y[i] += A[i][j] * x[j];
        }
    }
    return y;
}

// --------------------------------------------------------------------------
// Test: Tridiagonal system (m1=1, m2=1) with LU
// --------------------------------------------------------------------------
TEST(tridiagonal_lu) {
    // 4x4 tridiagonal:
    //  2 -1  0  0
    // -1  2 -1  0
    //  0 -1  2 -1
    //  0  0 -1  2
    const int n = 4, m1 = 1, m2 = 1;
    banded::BandMatrix A(n, m1, m2);

    // Dense reference
    std::vector<std::vector<double>> Ad(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        A(i, i) = 2.0;
        Ad[i][i] = 2.0;
        if (i > 0) { A(i, i - 1) = -1.0; Ad[i][i - 1] = -1.0; }
        if (i < n - 1) { A(i, i + 1) = -1.0; Ad[i][i + 1] = -1.0; }
    }

    std::vector<double> b = {1.0, 0.0, 0.0, 1.0};

    ASSERT_TRUE(banded::band_lu_decompose(A));
    auto x = banded::band_lu_solve(A, b);

    // Verify: rebuild A and check A*x = b
    // Expected solution for this system: x = [1, 1, 1, 1]
    // Check: 2*1 - 1*1 = 1 ✓, -1*1 + 2*1 - 1*1 = 0 ✓, etc.
    auto residual = band_matvec(n, m1, m2, Ad, x);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(residual[i], b[i], 1e-10);
    }
}

// --------------------------------------------------------------------------
// Test: Pentadiagonal system (m1=2, m2=2) with LU
// --------------------------------------------------------------------------
TEST(pentadiagonal_lu) {
    const int n = 6, m1 = 2, m2 = 2;
    banded::BandMatrix A(n, m1, m2);
    std::vector<std::vector<double>> Ad(n, std::vector<double>(n, 0.0));

    // Fill with a diagonally dominant band matrix
    for (int i = 0; i < n; i++) {
        A(i, i) = 6.0;
        Ad[i][i] = 6.0;
        if (i >= 1) { A(i, i - 1) = -1.0; Ad[i][i - 1] = -1.0; }
        if (i >= 2) { A(i, i - 2) = -0.5; Ad[i][i - 2] = -0.5; }
        if (i < n - 1) { A(i, i + 1) = -1.0; Ad[i][i + 1] = -1.0; }
        if (i < n - 2) { A(i, i + 2) = -0.5; Ad[i][i + 2] = -0.5; }
    }

    std::vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    ASSERT_TRUE(banded::band_lu_decompose(A));
    auto x = banded::band_lu_solve(A, b);

    auto residual = band_matvec(n, m1, m2, Ad, x);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(residual[i], b[i], 1e-10);
    }
}

// --------------------------------------------------------------------------
// Test: Tridiagonal system with QR
// --------------------------------------------------------------------------
TEST(tridiagonal_qr) {
    const int n = 4, m1 = 1, m2 = 1;
    banded::BandMatrix A(n, m1, m2);
    std::vector<std::vector<double>> Ad(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        A(i, i) = 2.0;
        Ad[i][i] = 2.0;
        if (i > 0) { A(i, i - 1) = -1.0; Ad[i][i - 1] = -1.0; }
        if (i < n - 1) { A(i, i + 1) = -1.0; Ad[i][i + 1] = -1.0; }
    }

    std::vector<double> b = {1.0, 0.0, 0.0, 1.0};

    auto qr = banded::band_qr_decompose(A);
    auto x = banded::band_qr_solve(qr, b);

    auto residual = band_matvec(n, m1, m2, Ad, x);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(residual[i], b[i], 1e-10);
    }
}

// --------------------------------------------------------------------------
// Test: Pentadiagonal system with QR
// --------------------------------------------------------------------------
TEST(pentadiagonal_qr) {
    const int n = 6, m1 = 2, m2 = 2;
    banded::BandMatrix A(n, m1, m2);
    std::vector<std::vector<double>> Ad(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        A(i, i) = 6.0;
        Ad[i][i] = 6.0;
        if (i >= 1) { A(i, i - 1) = -1.0; Ad[i][i - 1] = -1.0; }
        if (i >= 2) { A(i, i - 2) = -0.5; Ad[i][i - 2] = -0.5; }
        if (i < n - 1) { A(i, i + 1) = -1.0; Ad[i][i + 1] = -1.0; }
        if (i < n - 2) { A(i, i + 2) = -0.5; Ad[i][i + 2] = -0.5; }
    }

    std::vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    auto qr = banded::band_qr_decompose(A);
    auto x = banded::band_qr_solve(qr, b);

    auto residual = band_matvec(n, m1, m2, Ad, x);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(residual[i], b[i], 1e-10);
    }
}

// --------------------------------------------------------------------------
// Test: Larger random banded system — LU vs QR cross-check
// --------------------------------------------------------------------------
TEST(large_random_lu_vs_qr) {
    const int n = 100, m1 = 3, m2 = 4;
    std::vector<std::vector<double>> Ad(n, std::vector<double>(n, 0.0));

    // Build a diagonally dominant random band matrix
    std::srand(42);
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        int j_lo = std::max(0, i - m1);
        int j_hi = std::min(n - 1, i + m2);
        for (int j = j_lo; j <= j_hi; j++) {
            if (j != i) {
                double val = (std::rand() % 200 - 100) / 100.0; // [-1, 1]
                Ad[i][j] = val;
                row_sum += std::abs(val);
            }
        }
        Ad[i][i] = row_sum + 1.0; // diagonal dominance
    }

    // Random RHS
    std::vector<double> b(n);
    for (int i = 0; i < n; i++) {
        b[i] = (std::rand() % 200 - 100) / 10.0;
    }

    // LU solve
    banded::BandMatrix A_lu(n, m1, m2);
    for (int i = 0; i < n; i++) {
        int j_lo = std::max(0, i - m1);
        int j_hi = std::min(n - 1, i + m2);
        for (int j = j_lo; j <= j_hi; j++) {
            A_lu(i, j) = Ad[i][j];
        }
    }
    ASSERT_TRUE(banded::band_lu_decompose(A_lu));
    auto x_lu = banded::band_lu_solve(A_lu, b);

    // QR solve
    banded::BandMatrix A_qr(n, m1, m2);
    for (int i = 0; i < n; i++) {
        int j_lo = std::max(0, i - m1);
        int j_hi = std::min(n - 1, i + m2);
        for (int j = j_lo; j <= j_hi; j++) {
            A_qr(i, j) = Ad[i][j];
        }
    }
    auto qr = banded::band_qr_decompose(A_qr);
    auto x_qr = banded::band_qr_solve(qr, b);

    // Both solutions should match
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(x_lu[i], x_qr[i], 1e-8);
    }

    // Verify residual
    auto residual = band_matvec(n, m1, m2, Ad, x_lu);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(residual[i], b[i], 1e-8);
    }
}

// --------------------------------------------------------------------------
// Test: Performance — large tridiagonal system
// --------------------------------------------------------------------------
TEST(performance_tridiagonal) {
    const int n = 100000, m1 = 1, m2 = 1;
    banded::BandMatrix A(n, m1, m2);

    for (int i = 0; i < n; i++) {
        A(i, i) = 2.0;
        if (i > 0) A(i, i - 1) = -1.0;
        if (i < n - 1) A(i, i + 1) = -1.0;
    }

    std::vector<double> b(n, 1.0);

    auto t0 = std::chrono::high_resolution_clock::now();
    ASSERT_TRUE(banded::band_lu_decompose(A));
    auto x = banded::band_lu_solve(A, b);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("  LU solve n=%d, m1=%d, m2=%d: %.2f ms\n", n, m1, m2, ms);

    // Quick sanity check on solution
    ASSERT_TRUE(x.size() == static_cast<size_t>(n));
    ASSERT_TRUE(std::isfinite(x[0]));
    ASSERT_TRUE(std::isfinite(x[n - 1]));
}

int main() {
    // Tests run via static initialization above
    std::printf("\n=== Results: %d passed, %d failed ===\n",
                g_tests_passed, g_tests_failed);
    return g_tests_failed > 0 ? 1 : 0;
}
