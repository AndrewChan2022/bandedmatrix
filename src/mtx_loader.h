#pragma once

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mtx {

// COO triplet
struct Triplet {
    int row, col;
    double val;
};

// Loaded matrix info
struct MatrixInfo {
    int n;              // dimension (square)
    int m1;             // lower bandwidth
    int m2;             // upper bandwidth
    int nnz;            // number of nonzeros (including symmetric expansion)
    bool symmetric;
    std::string name;
    std::vector<Triplet> triplets; // all nonzeros (symmetric expanded)
};

// Load a Matrix Market (.mtx) file in coordinate format.
// Supports real symmetric and general square matrices.
// Returns MatrixInfo with computed bandwidth.
inline MatrixInfo load_mtx(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) throw std::runtime_error("Cannot open: " + path);

    MatrixInfo info;
    info.symmetric = false;
    info.n = 0;
    info.m1 = 0;
    info.m2 = 0;

    // Extract name from path
    auto slash = path.find_last_of("/\\");
    auto dot = path.find_last_of('.');
    if (slash != std::string::npos && dot != std::string::npos)
        info.name = path.substr(slash + 1, dot - slash - 1);
    else
        info.name = path;

    char line[1024];
    int nrows = 0, ncols = 0, nnz_file = 0;

    // Read header
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '%') {
            if (strstr(line, "symmetric")) info.symmetric = true;
            continue;
        }
        // First non-comment line: dimensions
        if (sscanf(line, "%d %d %d", &nrows, &ncols, &nnz_file) >= 3) break;
    }

    if (nrows != ncols) {
        fclose(f);
        throw std::runtime_error("Non-square matrix: " + path);
    }
    info.n = nrows;

    // Reserve space
    info.triplets.reserve(info.symmetric ? nnz_file * 2 : nnz_file);

    // Read triplets
    int r, c;
    double v;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '%') continue;
        v = 0.0;
        int nread = sscanf(line, "%d %d %lf", &r, &c, &v);
        if (nread >= 2) {
            // MTX is 1-indexed
            r--; c--;
            if (nread < 3) v = 1.0; // pattern matrix

            info.triplets.push_back({r, c, v});
            int bw = r - c;
            if (bw > info.m1) info.m1 = bw;
            if (-bw > info.m2) info.m2 = -bw;

            // Add symmetric counterpart
            if (info.symmetric && r != c) {
                info.triplets.push_back({c, r, v});
                if (-bw > info.m1) info.m1 = -bw; // swap role
                if (bw > info.m2) info.m2 = bw;
            }
        }
    }

    fclose(f);
    info.nnz = static_cast<int>(info.triplets.size());
    return info;
}

} // namespace mtx
