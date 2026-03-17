#include "../src/banded_matrix.h"
#include "../src/mtx_loader.h"
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
#include <string>
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
// Result
// --------------------------------------------------------------------------
struct BenchResult {
    const char* method;
    double decompose_ms;
    double solve_ms;
    double total_ms;
    double residual;
};

static void print_header() {
    std::printf("    %-24s %12s %12s %12s %14s\n",
                "Method", "Decomp(ms)", "Solve(ms)", "Total(ms)", "Residual");
    std::printf("    %-24s %12s %12s %12s %14s\n",
                "------", "----------", "---------", "---------", "--------");
}

static void print_result(const BenchResult& r) {
    std::printf("    %-24s %12.3f %12.3f %12.3f %14.2e\n",
                r.method, r.decompose_ms, r.solve_ms, r.total_ms, r.residual);
}

// --------------------------------------------------------------------------
// Benchmark a single matrix
// --------------------------------------------------------------------------
static void benchmark_matrix(const std::string& mtx_path) {
    // Load matrix
    mtx::MatrixInfo info;
    try {
        info = mtx::load_mtx(mtx_path);
    } catch (const std::exception& e) {
        std::printf("  SKIP %s: %s\n\n", mtx_path.c_str(), e.what());
        return;
    }

    int n = info.n;
    int m1 = info.m1;
    int m2 = info.m2;
    int bw = m1 + m2 + 1;

    std::printf("  Matrix: %s\n", info.name.c_str());
    std::printf("    N = %d, nnz = %d, symmetric = %s\n", n, info.nnz, info.symmetric ? "yes" : "no");
    std::printf("    bandwidth: m1 = %d, m2 = %d, total = %d (%.1f%% of N)\n",
                m1, m2, bw, 100.0 * bw / n);
    std::printf("    band storage: %.1f KB vs dense: %.1f KB (%.1fx compression)\n",
                (double)n * bw * 8 / 1024.0,
                (double)n * n * 8 / 1024.0,
                (double)n / bw);
    std::printf("\n");

    // Build Eigen sparse matrix
    std::vector<Eigen::Triplet<double>> eigen_triplets;
    eigen_triplets.reserve(info.nnz);
    for (const auto& t : info.triplets) {
        eigen_triplets.emplace_back(t.row, t.col, t.val);
    }
    Eigen::SparseMatrix<double> As(n, n);
    As.setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());
    As.makeCompressed();

    // Build our banded matrix
    banded::BandMatrix A_band(n, m1, m2);
    for (const auto& t : info.triplets) {
        A_band(t.row, t.col) = t.val;
    }

    // Check diagonal dominance; if not, add diagonal shift for stability
    // (many structural matrices are SPD but not diagonally dominant)
    // We solve A*x = b, so A must be nonsingular. SPD is fine for LU.

    // Generate RHS: b = A * ones(n) so we know exact solution x = ones(n)
    Eigen::VectorXd x_exact = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd b_eigen = As * x_exact;
    std::vector<double> b_std(n);
    for (int i = 0; i < n; i++) b_std[i] = b_eigen(i);

    // Compute residual helper (using sparse matvec)
    auto compute_residual = [&](const Eigen::VectorXd& x) -> double {
        return (As * x - b_eigen).cwiseAbs().maxCoeff();
    };

    auto compute_residual_std = [&](const std::vector<double>& x) -> double {
        Eigen::VectorXd xe(n);
        for (int i = 0; i < n; i++) xe(i) = x[i];
        return compute_residual(xe);
    };

    print_header();

    // --- Our Banded LU ---
    {
        banded::BandMatrix A_copy = A_band;
        BenchResult r;
        r.method = "Our Banded LU";

        Timer t1;
        bool ok = banded::band_lu_decompose(A_copy);
        r.decompose_ms = t1.elapsed_ms();

        if (ok) {
            Timer t2;
            auto x = banded::band_lu_solve(A_copy, b_std);
            r.solve_ms = t2.elapsed_ms();
            r.total_ms = r.decompose_ms + r.solve_ms;
            r.residual = compute_residual_std(x);
        } else {
            r.solve_ms = 0;
            r.total_ms = r.decompose_ms;
            r.residual = -1.0;
        }
        print_result(r);
    }

    // --- Our Banded QR ---
    {
        BenchResult r;
        r.method = "Our Banded QR";

        Timer t1;
        auto qr = banded::band_qr_decompose(A_band);
        r.decompose_ms = t1.elapsed_ms();

        try {
            Timer t2;
            auto x = banded::band_qr_solve(qr, b_std);
            r.solve_ms = t2.elapsed_ms();
            r.total_ms = r.decompose_ms + r.solve_ms;
            r.residual = compute_residual_std(x);
        } catch (...) {
            r.solve_ms = 0;
            r.total_ms = r.decompose_ms;
            r.residual = -1.0;
        }
        print_result(r);
    }

    // --- Eigen Sparse LU ---
    {
        BenchResult r;
        r.method = "Eigen Sparse LU";

        Timer t1;
        Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
        solver.analyzePattern(As);
        solver.factorize(As);
        r.decompose_ms = t1.elapsed_ms();

        if (solver.info() == Eigen::Success) {
            Timer t2;
            Eigen::VectorXd x = solver.solve(b_eigen);
            r.solve_ms = t2.elapsed_ms();
            r.total_ms = r.decompose_ms + r.solve_ms;
            r.residual = compute_residual(x);
        } else {
            r.solve_ms = 0;
            r.total_ms = r.decompose_ms;
            r.residual = -1.0;
        }
        print_result(r);
    }

    // --- Eigen Sparse QR ---
    {
        BenchResult r;
        r.method = "Eigen Sparse QR";

        Timer t1;
        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
        solver.analyzePattern(As);
        solver.factorize(As);
        r.decompose_ms = t1.elapsed_ms();

        if (solver.info() == Eigen::Success) {
            Timer t2;
            Eigen::VectorXd x = solver.solve(b_eigen);
            r.solve_ms = t2.elapsed_ms();
            r.total_ms = r.decompose_ms + r.solve_ms;
            r.residual = compute_residual(x);
        } else {
            r.solve_ms = 0;
            r.total_ms = r.decompose_ms;
            r.residual = -1.0;
        }
        print_result(r);
    }

    std::printf("\n");
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::printf("=== Real-World Banded Matrix Benchmark ===\n");
    std::printf("Our Banded LU/QR vs Eigen Sparse LU/QR\n");
    std::printf("Matrices from SuiteSparse Matrix Collection (HB group)\n\n");

    // If command-line args given, use those as .mtx paths
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            benchmark_matrix(argv[i]);
        }
        return 0;
    }

    // Default: benchmark all matrices in data/
    const char* matrices[] = {
        "data/nos1/nos1.mtx",         // n=237,  bw=4,  beam structure
        "data/bcsstk03/bcsstk03.mtx", // n=112,  bw=7,  structural
        "data/nos4/nos4.mtx",         // n=100,  bw=13, beam structure
        "data/bcsstk20/bcsstk20.mtx", // n=485,  bw=20, structural
        "data/nos6/nos6.mtx",         // n=675,  bw=30, beam structure
        "data/nos3/nos3.mtx",         // n=960,  bw=43, beam structure
        "data/nos7/nos7.mtx",         // n=729,  bw=81, beam structure
        "data/bcsstk21/bcsstk21.mtx", // n=3600, bw=125, structural
    };

    for (const char* path : matrices) {
        benchmark_matrix(path);
    }

    std::printf("Done.\n");
    return 0;
}
