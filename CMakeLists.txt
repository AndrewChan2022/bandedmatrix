cmake_minimum_required(VERSION 3.14)
project(bandedmatrix LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Libraries
add_library(banded_matrix src/banded_matrix.cpp)
target_include_directories(banded_matrix PUBLIC src)

add_library(dense_matrix src/dense_matrix.cpp)
target_include_directories(dense_matrix PUBLIC src)

# Tests
add_executable(test_banded_matrix test/test_banded_matrix.cpp)
target_link_libraries(test_banded_matrix PRIVATE banded_matrix)

# Benchmark: dense vs banded
add_executable(benchmark test/benchmark.cpp)
target_link_libraries(benchmark PRIVATE banded_matrix dense_matrix)

# Benchmark with Eigen3
find_package(Eigen3 3.4 QUIET HINTS "E:/local/eigen3/3.4.90" "E:/local/Eigen")
if(Eigen3_FOUND)
    add_executable(benchmark_eigen test/benchmark_eigen.cpp)
    target_link_libraries(benchmark_eigen PRIVATE banded_matrix dense_matrix Eigen3::Eigen)
    message(STATUS "Eigen3 found: ${Eigen3_DIR} — benchmark_eigen target enabled")
else()
    message(STATUS "Eigen3 not found — benchmark_eigen target skipped")
endif()

# Demo
add_executable(demo demo/demo.cpp)
target_link_libraries(demo PRIVATE banded_matrix)

# Release build for performance benchmarks
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
