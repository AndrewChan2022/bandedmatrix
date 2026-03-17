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

# Demo
add_executable(demo demo/demo.cpp)
target_link_libraries(demo PRIVATE banded_matrix)

# Release build for performance benchmarks
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
