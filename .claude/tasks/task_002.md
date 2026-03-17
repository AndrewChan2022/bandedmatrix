

I want to benchmark the banded_matrix against eigen sparse with real world dataset.

info in this, please do it.  you need just implement banded LU and banded QR, ignore others.


test dataset store to data/

```txt

我来帮你研究 Eigen 矩阵求解相关的数据集。Eigen 是一个高性能的 C++ 线性代数库，广泛用于矩阵运算、求解线性方程组、特征值计算等场景。让我搜索相关的基准测试数据集和研究资源。
基于搜索结果，我为你整理 Eigen 矩阵求解数据集 的完整资源指南。这些数据集广泛用于测试和基准评估 Eigen 的各类求解器（稠密/稀疏、直接/迭代）。

---

1. 标准公开数据集

Matrix Market (NIST)
最权威的稀疏矩阵数据集，Eigen 官方文档和大量研究论文使用其进行基准测试：

- 格式: `.mtx` 文件（Matrix Market 格式）
- Eigen 读取: 使用 `<unsupported/Eigen/SparseExtra>` 中的 `Eigen::loadMarket()`
- 覆盖领域: 结构工程、流体力学、电磁学、经济模型等
- 规模: 从数百阶到数百万阶稀疏矩阵

```cpp
#include <unsupported/Eigen/SparseExtra>
Eigen::SparseMatrix<double> A;
Eigen::loadMarket(A, "matrix.mtx");
```

SuiteSparse Matrix Collection (原 UF Sparse Matrix Collection)
- 超过 2800 个真实世界稀疏矩阵
- 包含 Tim Davis 维护的结构性、统计性、机器学习等领域矩阵
- 网站: https://sparse.tamu.edu/

---

2. Eigen 官方测试数据集

Eigen 自带超过 500 个可执行测试文件 ，涵盖:

测试类型	说明	
BLAS 测试套件	与标准 BLAS 库对比	
LAPACK 测试子集	稠密矩阵分解、特征值求解	
稀疏求解器测试	SparseLU、SparseQR、SimplicialLDLT/LLT、BiCGSTAB、ConjugateGradient 等	

---

3. 学术研究常用数据集

材料科学矩阵集合 (Matrix Collection in Material Science)
东京大学 ISSP 开发，专门用于稀疏矩阵求解器性能研究 ：
- 网址: https://isspns-gilab.issp.u-tokyo.ac.jp/takeohoshi/matrix-collection-in-material-science/
- 适用: 电子结构计算、量子力学模拟

机器人学 KKT 矩阵数据集
Pinocchio 3 库使用的基准测试 ：
- 机器人约束动力学产生的 KKT 矩阵
- 用于测试 Cholesky 分解（`Eigen::SimplicialLDLT` vs `Eigen::LDLT`）
- 规模: 从 UR5 机械臂到人形机器人 Talos/iCub

---

4. 求解器分类与适用数据集

根据 Eigen 官方文档 ，求解器与典型数据集匹配：

求解器	矩阵类型	推荐数据集特征	适用规模	
`SimplicialLDLT`	SPD（对称正定）	2D Poisson 方程、有限元刚度矩阵	中小规模，极稀疏	
`SimplicialLLT`	SPD	同上，但 LDLT 通常更优	中小规模	
`SparseLU`	方阵（一般）	非对称、不规则稀疏模式	小到大规模	
`SparseQR`	任意（含矩形）	最小二乘问题、秩亏矩阵	中小规模	
`BiCGSTAB` + ILUT	一般方阵	非对称、需要预处理	大规模	
`ConjugateGradient` + Diagonal	SPD	对称正定、对角预处理有效	大规模	

---

5. 性能基准测试代码模板

基于搜索结果  的完整测试框架：

```cpp
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <ctime>

using SpMatrix = Eigen::SparseMatrix<double>;
using Matrix = Eigen::MatrixXd;

int main() {
    // 加载 Matrix Market 数据集
    SpMatrix A, B;
    Eigen::loadMarket(A, "A.mtx");
    Eigen::loadMarket(B, "B.mtx");  // 多右端项
    
    std::clock_t start;
    
    // 1. 迭代求解器: LeastSquaresConjugateGradient
    {
        start = std::clock();
        Eigen::LeastSquaresConjugateGradient<SpMatrix> solver;
        solver.setTolerance(1e-10);
        solver.compute(A);
        Matrix X = solver.solve(B);  // 注意: 显式转为 Matrix 避免性能损失
        std::cout << "LSCG: " << (std::clock() - start) / (double)CLOCKS_PER_SEC << "s\n";
    }
    
    // 2. 直接求解器: SimplicialLDLT (SPD 矩阵)
    {
        start = std::clock();
        Eigen::SimplicialLDLT<SpMatrix> solver;
        solver.compute(A.transpose() * A);  // 正规方程
        Matrix X = solver.solve(A.transpose() * B);
        std::cout << "LDLT: " << (std::clock() - start) / (double)CLOCKS_PER_SEC << "s\n";
    }
    
    // 3. 稀疏 LU (一般矩阵)
    {
        start = std::clock();
        Eigen::SparseLU<SpMatrix> solver;
        solver.analyzePattern(A);   // 可复用模式分析
        solver.factorize(A);        // 仅数值分解
        Matrix X = solver.solve(B);
        std::cout << "LU: " << (std::clock() - start) / (double)CLOCKS_PER_SEC << "s\n";
    }
    
    return 0;
}
```

---

6. 关键注意事项

1. 右端项格式: 当右端项是稀疏矩阵时，显式转为稠密 Matrix 可显著提升性能（15.4s → 1.4s）[17^]
2. 多线程: 使用 `Eigen::initParallel()` 和 `Eigen::setNbThreads()` 启用 OpenMP 并行
3. 预处理: 迭代求解器必须配合预处理器（如 `Eigen::IncompleteLUT`）[15^]
4. 模式复用: 对相同稀疏结构的多组求解，分离 `analyzePattern()` 和 `factorize()` 

---

7. 推荐获取路径

需求场景	推荐数据集	获取方式	
通用稀疏求解器测试	Matrix Market	https://math.nist.gov/MatrixMarket/	
大规模科学计算	SuiteSparse Collection	https://sparse.tamu.edu/	
材料/物理仿真	ISSP Material Science Collection	GitLab ISSP	
机器人/多体动力学	Pinocchio URDF/SDF 模型	GitHub stack-of-tasks/pinocchio	

需要针对特定领域（如 CAE/CFD）的矩阵数据集推荐吗？

```

