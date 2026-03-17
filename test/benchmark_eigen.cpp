#include "../src/banded_matrix.h"
#include "../src/dense_matrix.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <Eigen/OrderingMethods>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

// --------------------------------------------------------------------------
// Timer
// --------------------------------------------------------------------------
struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    Timer() : t0(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// --------------------------------------------------------------------------
// Result printing
// --------------------------------------------------------------------------
struct BenchResult {
    const char* method;
    int n;
    int bandwidth;
    double decompose_ms;
    double solve_ms;
    double total_ms;
    double residual;
};

static void print_header() {
    std::printf("  %-24s %10s %6s %12s %12s %12s %14s\n",
                "Method", "N", "Band", "Decomp(ms)", "Solve(ms)", "Total(ms)", "Residual");
    std::printf("  %-24s %10s %6s %12s %12s %12s %14s\n",
                "------", "---", "----", "----------", "---------", "---------", "--------");
}

static void print_result(const BenchResult& r) {
    std::printf("  %-24s %10d %6d %12.3f %12.3f %12.3f %14.2e\n",
                r.method, r.n, r.bandwidth,
                r.decompose_ms, r.solve_ms, r.total_ms, r.residual);
}

// --------------------------------------------------------------------------
// Build test data: diagonally dominant banded matrix (same for all methods)
// --------------------------------------------------------------------------
struct TestData {
    int n, m1, m2;
    // Dense format (for Eigen Dense and our dense solver)
    Eigen::MatrixXd A_dense;
    Eigen::VectorXd b_eigen;
    std::vector<double> b_std;
};

static TestData make_test_data(int n, int band) {
    TestData td;
    td.n = n;
    td.m1 = band;
    td.m2 = band;

    td.A_dense = Eigen::MatrixXd::Zero(n, n);
    td.b_eigen.resize(n);
    td.b_std.resize(n);

    std::srand(12345);
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        int j_lo = std::max(0, i - band);
        int j_hi = std::min(n - 1, i + band);
        for (int j = j_lo; j <= j_hi; j++) {
            if (j != i) {
                double val = (std::rand() % 200 - 100) / 100.0;
                td.A_dense(i, j) = val;
                row_sum += std::abs(val);
            }
        }
        td.A_dense(i, i) = row_sum + 1.0;
    }

    for (int i = 0; i < n; i++) {
        double val = (std::rand() % 200 - 100) / 10.0;
        td.b_eigen(i) = val;
        td.b_std[i] = val;
    }

    return td;
}

// --------------------------------------------------------------------------
// Our Banded LU
// --------------------------------------------------------------------------
static BenchResult bench_our_banded_lu(const TestData& td) {
    int n = td.n, m1 = td.m1, m2 = td.m2;
    banded::BandMatrix A(n, m1, m2);
    for (int i = 0; i < n; i++) {
        int j_lo = std::max(0, i - m1);
        int j_hi = std::min(n - 1, i + m2);
        for (int j = j_lo; j <= j_hi; j++)
            A(i, j) = td.A_dense(i, j);
    }

    BenchResult r;
    r.method = "Our Banded LU";
    r.n = n;
    r.bandwidth = m1;

    Timer t1;
    banded::band_lu_decompose(A);
    r.decompose_ms = t1.elapsed_ms();

    Timer t2;
    auto x = banded::band_lu_solve(A, td.b_std);
    r.solve_ms = t2.elapsed_ms();
    r.total_ms = r.decompose_ms + r.solve_ms;

    // Residual
    Eigen::VectorXd xe(n);
    for (int i = 0; i < n; i++) xe(i) = x[i];
    r.residual = (td.A_dense * xe - td.b_eigen).cwiseAbs().maxCoeff();
    return r;
}

// --------------------------------------------------------------------------
// Our Banded QR
// --------------------------------------------------------------------------
static BenchResult bench_our_banded_qr(const TestData& td) {
    int n = td.n, m1 = td.m1, m2 = td.m2;
    banded::BandMatrix A(n, m1, m2);
    for (int i = 0; i < n; i++) {
        int j_lo = std::max(0, i - m1);
        int j_hi = std::min(n - 1, i + m2);
        for (int j = j_lo; j <= j_hi; j++)
            A(i, j) = td.A_dense(i, j);
    }

    BenchResult r;
    r.method = "Our Banded QR";
    r.n = n;
    r.bandwidth = m1;

    Timer t1;
    auto qr = banded::band_qr_decompose(A);
    r.decompose_ms = t1.elapsed_ms();

    Timer t2;
    auto x = banded::band_qr_solve(qr, td.b_std);
    r.solve_ms = t2.elapsed_ms();
    r.total_ms = r.decompose_ms + r.solve_ms;

    Eigen::VectorXd xe(n);
    for (int i = 0; i < n; i++) xe(i) = x[i];
    r.residual = (td.A_dense * xe - td.b_eigen).cwiseAbs().maxCoeff();
    return r;
}

// --------------------------------------------------------------------------
// Our Dense LU
// --------------------------------------------------------------------------
static BenchResult bench_our_dense_lu(const TestData& td) {
    int n = td.n;
    dense::DenseMatrix A(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A(i, j) = td.A_dense(i, j);

    BenchResult r;
    r.method = "Our Dense LU";
    r.n = n;
    r.bandwidth = td.m1;

    Timer t1;
    dense::dense_lu_decompose(A);
    r.decompose_ms = t1.elapsed_ms();

    Timer t2;
    auto x = dense::dense_lu_solve(A, td.b_std);
    r.solve_ms = t2.elapsed_ms();
    r.total_ms = r.decompose_ms + r.solve_ms;

    Eigen::VectorXd xe(n);
    for (int i = 0; i < n; i++) xe(i) = x[i];
    r.residual = (td.A_dense * xe - td.b_eigen).cwiseAbs().maxCoeff();
    return r;
}

// --------------------------------------------------------------------------
// Our Dense QR
// --------------------------------------------------------------------------
static BenchResult bench_our_dense_qr(const TestData& td) {
    int n = td.n;
    dense::DenseMatrix A(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A(i, j) = td.A_dense(i, j);

    BenchResult r;
    r.method = "Our Dense QR";
    r.n = n;
    r.bandwidth = td.m1;

    Timer t1;
    auto qr = dense::dense_qr_decompose(A);
    r.decompose_ms = t1.elapsed_ms();

    Timer t2;
    auto x = dense::dense_qr_solve(qr, td.b_std);
    r.solve_ms = t2.elapsed_ms();
    r.total_ms = r.decompose_ms + r.solve_ms;

    Eigen::VectorXd xe(n);
    for (int i = 0; i < n; i++) xe(i) = x[i];
    r.residual = (td.A_dense * xe - td.b_eigen).cwiseAbs().maxCoeff();
    return r;
}

// --------------------------------------------------------------------------
// Eigen Dense PartialPivLU
// --------------------------------------------------------------------------
static BenchResult bench_eigen_dense_lu(const TestData& td) {
    BenchResult r;
    r.method = "Eigen Dense LU";
    r.n = td.n;
    r.bandwidth = td.m1;

    Timer t1;
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(td.A_dense);
    r.decompose_ms = t1.elapsed_ms();

    Timer t2;
    Eigen::VectorXd x = lu.solve(td.b_eigen);
    r.solve_ms = t2.elapsed_ms();
    r.total_ms = r.decompose_ms + r.solve_ms;

    r.residual = (td.A_dense * x - td.b_eigen).cwiseAbs().maxCoeff();
    return r;
}

// --------------------------------------------------------------------------
// Eigen Dense HouseholderQR
// --------------------------------------------------------------------------
static BenchResult bench_eigen_dense_qr(const TestData& td) {
    BenchResult r;
    r.method = "Eigen Dense QR";
    r.n = td.n;
    r.bandwidth = td.m1;

    Timer t1;
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(td.A_dense);
    r.decompose_ms = t1.elapsed_ms();

    Timer t2;
    Eigen::VectorXd x = qr.solve(td.b_eigen);
    r.solve_ms = t2.elapsed_ms();
    r.total_ms = r.decompose_ms + r.solve_ms;

    r.residual = (td.A_dense * x - td.b_eigen).cwiseAbs().maxCoeff();
    return r;
}

// --------------------------------------------------------------------------
// Eigen Sparse LU (exploits sparsity pattern from banded structure)
// --------------------------------------------------------------------------
static BenchResult bench_eigen_sparse_lu(const TestData& td) {
    int n = td.n, m1 = td.m1, m2 = td.m2;

    // Build sparse matrix
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<size_t>(n) * (m1 + m2 + 1));
    for (int i = 0; i < n; i++) {
        int j_lo = std::max(0, i - m1);
        int j_hi = std::min(n - 1, i + m2);
        for (int j = j_lo; j <= j_hi; j++) {
            double val = td.A_dense(i, j);
            if (val != 0.0) triplets.emplace_back(i, j, val);
        }
    }
    Eigen::SparseMatrix<double> As(n, n);
    As.setFromTriplets(triplets.begin(), triplets.end());
    As.makeCompressed();

    BenchResult r;
    r.method = "Eigen Sparse LU";
    r.n = n;
    r.bandwidth = m1;

    Timer t1;
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    solver.analyzePattern(As);
    solver.factorize(As);
    r.decompose_ms = t1.elapsed_ms();

    Timer t2;
    Eigen::VectorXd x = solver.solve(td.b_eigen);
    r.solve_ms = t2.elapsed_ms();
    r.total_ms = r.decompose_ms + r.solve_ms;

    r.residual = (td.A_dense * x - td.b_eigen).cwiseAbs().maxCoeff();
    return r;
}

// --------------------------------------------------------------------------
// Eigen Sparse QR
// --------------------------------------------------------------------------
static BenchResult bench_eigen_sparse_qr(const TestData& td) {
    int n = td.n, m1 = td.m1, m2 = td.m2;

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<size_t>(n) * (m1 + m2 + 1));
    for (int i = 0; i < n; i++) {
        int j_lo = std::max(0, i - m1);
        int j_hi = std::min(n - 1, i + m2);
        for (int j = j_lo; j <= j_hi; j++) {
            double val = td.A_dense(i, j);
            if (val != 0.0) triplets.emplace_back(i, j, val);
        }
    }
    Eigen::SparseMatrix<double> As(n, n);
    As.setFromTriplets(triplets.begin(), triplets.end());
    As.makeCompressed();

    BenchResult r;
    r.method = "Eigen Sparse QR";
    r.n = n;
    r.bandwidth = m1;

    Timer t1;
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    solver.analyzePattern(As);
    solver.factorize(As);
    r.decompose_ms = t1.elapsed_ms();

    Timer t2;
    Eigen::VectorXd x = solver.solve(td.b_eigen);
    r.solve_ms = t2.elapsed_ms();
    r.total_ms = r.decompose_ms + r.solve_ms;

    r.residual = (td.A_dense * x - td.b_eigen).cwiseAbs().maxCoeff();
    return r;
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------
int main() {
    std::printf("=== Banded Matrix Solver Benchmark: Ours vs Eigen3 ===\n\n");
    std::printf("Eigen version: %d.%d.%d\n", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
    std::printf("All matrices: diagonally dominant, band = 3 (bandwidth = 7)\n\n");

    const int band = 3;

    // ---- (a) Small: N = 10 ----
    {
        const int n = 10;
        auto td = make_test_data(n, band);
        std::printf("--- (a) Small: N = %d ---\n\n", n);
        print_header();
        print_result(bench_our_banded_lu(td));
        print_result(bench_our_banded_qr(td));
        print_result(bench_our_dense_lu(td));
        print_result(bench_our_dense_qr(td));
        print_result(bench_eigen_dense_lu(td));
        print_result(bench_eigen_dense_qr(td));
        print_result(bench_eigen_sparse_lu(td));
        print_result(bench_eigen_sparse_qr(td));
        std::printf("\n");
    }

    // ---- (b) Middle: N = 1000 ----
    {
        const int n = 1000;
        auto td = make_test_data(n, band);
        std::printf("--- (b) Middle: N = %d ---\n\n", n);
        print_header();
        print_result(bench_our_banded_lu(td));
        print_result(bench_our_banded_qr(td));
        print_result(bench_our_dense_lu(td));
        print_result(bench_our_dense_qr(td));
        print_result(bench_eigen_dense_lu(td));
        print_result(bench_eigen_dense_qr(td));
        print_result(bench_eigen_sparse_lu(td));
        print_result(bench_eigen_sparse_qr(td));
        std::printf("\n");
    }

    // ---- (d) Large: N = 1,000,000 ----
    // Dense methods (ours and Eigen) are infeasible. Only banded/sparse.
    {
        const int n = 1000000;
        std::printf("--- (d) Large: N = %d ---\n", n);
        std::printf("NOTE: Dense methods skipped (N^2 = 10^12 doubles = 8 TB RAM)\n\n");

        // For large N, we can't store td.A_dense. Build sparse-only test data.
        std::srand(12345);

        // Build banded matrix for our solver
        banded::BandMatrix A_band(n, band, band);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(static_cast<size_t>(n) * (2 * band + 1));

        for (int i = 0; i < n; i++) {
            double row_sum = 0.0;
            int j_lo = std::max(0, i - band);
            int j_hi = std::min(n - 1, i + band);
            for (int j = j_lo; j <= j_hi; j++) {
                if (j != i) {
                    double val = (std::rand() % 200 - 100) / 100.0;
                    A_band(i, j) = val;
                    triplets.emplace_back(i, j, val);
                    row_sum += std::abs(val);
                }
            }
            double diag = row_sum + 1.0;
            A_band(i, i) = diag;
            triplets.emplace_back(i, i, diag);
        }

        // Save copy of band data for our solver residual check
        std::vector<double> au_copy = A_band.au;
        int mm = band + band + 1;

        std::vector<double> b_std(n);
        Eigen::VectorXd b_eigen(n);
        for (int i = 0; i < n; i++) {
            double val = (std::rand() % 200 - 100) / 10.0;
            b_std[i] = val;
            b_eigen(i) = val;
        }

        // Eigen sparse matrix
        Eigen::SparseMatrix<double> As(n, n);
        As.setFromTriplets(triplets.begin(), triplets.end());
        As.makeCompressed();
        triplets.clear();
        triplets.shrink_to_fit();

        print_header();

        // Our Banded LU
        {
            banded::BandMatrix A_lu = A_band;
            BenchResult r;
            r.method = "Our Banded LU";
            r.n = n;
            r.bandwidth = band;

            Timer t1;
            banded::band_lu_decompose(A_lu);
            r.decompose_ms = t1.elapsed_ms();

            Timer t2;
            auto x = banded::band_lu_solve(A_lu, b_std);
            r.solve_ms = t2.elapsed_ms();
            r.total_ms = r.decompose_ms + r.solve_ms;

            // Compute residual via banded matvec (can't use dense A_dense)
            double max_res = 0.0;
            for (int i = 0; i < n; i++) {
                double row_val = 0.0;
                int j_lo = std::max(0, i - band);
                int j_hi = std::min(n - 1, i + band);
                for (int j = j_lo; j <= j_hi; j++)
                    row_val += au_copy[static_cast<size_t>(i) * mm + (j - i + band)] * x[j];
                max_res = std::max(max_res, std::abs(row_val - b_std[i]));
            }
            r.residual = max_res;
            print_result(r);
        }

        // Our Banded QR
        {
            BenchResult r;
            r.method = "Our Banded QR";
            r.n = n;
            r.bandwidth = band;

            Timer t1;
            auto qr = banded::band_qr_decompose(A_band);
            r.decompose_ms = t1.elapsed_ms();

            Timer t2;
            auto x = banded::band_qr_solve(qr, b_std);
            r.solve_ms = t2.elapsed_ms();
            r.total_ms = r.decompose_ms + r.solve_ms;

            double max_res = 0.0;
            for (int i = 0; i < n; i++) {
                double row_val = 0.0;
                int j_lo = std::max(0, i - band);
                int j_hi = std::min(n - 1, i + band);
                for (int j = j_lo; j <= j_hi; j++)
                    row_val += au_copy[static_cast<size_t>(i) * mm + (j - i + band)] * x[j];
                max_res = std::max(max_res, std::abs(row_val - b_std[i]));
            }
            r.residual = max_res;
            print_result(r);
        }

        // Eigen Sparse LU
        {
            BenchResult r;
            r.method = "Eigen Sparse LU";
            r.n = n;
            r.bandwidth = band;

            Timer t1;
            Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
            solver.analyzePattern(As);
            solver.factorize(As);
            r.decompose_ms = t1.elapsed_ms();

            Timer t2;
            Eigen::VectorXd x = solver.solve(b_eigen);
            r.solve_ms = t2.elapsed_ms();
            r.total_ms = r.decompose_ms + r.solve_ms;

            double max_res = 0.0;
            for (int i = 0; i < n; i++) {
                double row_val = 0.0;
                int j_lo = std::max(0, i - band);
                int j_hi = std::min(n - 1, i + band);
                for (int j = j_lo; j <= j_hi; j++)
                    row_val += au_copy[static_cast<size_t>(i) * mm + (j - i + band)] * x[j];
                max_res = std::max(max_res, std::abs(row_val - b_std[i]));
            }
            r.residual = max_res;
            print_result(r);
        }

        // Eigen Sparse QR
        {
            BenchResult r;
            r.method = "Eigen Sparse QR";
            r.n = n;
            r.bandwidth = band;

            Timer t1;
            Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
            solver.analyzePattern(As);
            solver.factorize(As);
            r.decompose_ms = t1.elapsed_ms();

            Timer t2;
            Eigen::VectorXd x = solver.solve(b_eigen);
            r.solve_ms = t2.elapsed_ms();
            r.total_ms = r.decompose_ms + r.solve_ms;

            double max_res = 0.0;
            for (int i = 0; i < n; i++) {
                double row_val = 0.0;
                int j_lo = std::max(0, i - band);
                int j_hi = std::min(n - 1, i + band);
                for (int j = j_lo; j <= j_hi; j++)
                    row_val += au_copy[static_cast<size_t>(i) * mm + (j - i + band)] * x[j];
                max_res = std::max(max_res, std::abs(row_val - b_std[i]));
            }
            r.residual = max_res;
            print_result(r);
        }

        std::printf("\n");
    }

    std::printf("Done.\n");
    return 0;
}
