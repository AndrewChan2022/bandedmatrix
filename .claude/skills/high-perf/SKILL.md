---
name: high-perf
description: optimize for high performance, use low-level APIs, and write efficient code. Use when the user asks for "optimize for performance", "make it faster", or when working on performance-critical code.

---

When optimizing code, always include:

1. **data locality**: Organize data in memory to minimize cache misses and improve access speed. never use fragment memory layout, like vector of vector, hash map, etc. prefer CSR foramt, i.e. Compressed Sparse Row format, to store large dataset with shared elements, for example.
2. **prefer prefix scan than hash map for big data** For large datasets, linear scans can be more cache-friendly and faster than hash maps due to better data locality and reduced overhead. for example, for 100 million triangles with shared edges, using a hash map to find shared edges can lead to significant cache misses and performance degradation, while we can split the edges into virtual buckets based on their vertex indices, first scan per vertex bucket size, then put all bucket edges into a contiguous memory which we can CSR format i.e. Compressed Sparse Row format, then we a. scan per bucket size b. scatter edges to bucket c. local sort for each bucket d. dedup and assign global edge id. This approach minimizes cache misses and can be significantly faster than using a hash map for large datasets.
3. **Minimize branching**: Reduce conditional statements inside performance-critical loops to avoid pipeline stalls.
4. **Use efficient algorithms**: Choose algorithms with lower time complexity and optimize for the specific data size and access patterns. For example, to solve dense matrix, time is O(N^2), but if 
we know it is banded matrix, we can use banded matrix solver with time O(N * bandwidth), i.e. almost O(N) for small bandwith.
5. **Leverage SIMD and parallelism**: Utilize vectorized instructions and multi-threading to process multiple data elements simultaneously.
6. **Profile and measure**: Always use profiling tools to identify bottlenecks and validate performance improvements.
7. **Use accelerate structure** for geometric queries, for example, BVH for ray tracing, spatial hashing for collision detection, kdtree for knn, octree for sparse 3D data, etc.
8. **Precompute and reuse**: Precompute values that are expensive to calculate and reuse them when possible to avoid redundant computations.
9. **Avoid unnecessary compute**: Eliminate computations that do not contribute to the final result, such as redundant calculations or processing of irrelevant data.
For example, if we want compute volume of region volume cells that only cell number less than 100, 
we should exclude region that has cells number more than 100 before compute volume, instead of compute volume for all region then filter by cell number. This can save a lot of unnecessary computation and improve performance significantly, especially the bigger region cost more time to compute volume.
10. **multiple-scale pattern**: For problems that can be solved at multiple scales, start with a coarse approximation to quickly identify areas of interest, then refine the solution in those areas. For example, to blur a large image, we can build pyramid of the image with different levels of detail, each level with a blur, the final small image is blurred result. the total time is O(N) for the whole image, but we can get a blurred image with much less time than directly blur the original image with large kernel size.
