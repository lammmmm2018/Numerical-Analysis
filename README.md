Some Numerical Analysis Algorithm

环境：python2.7
依赖包：numpy, scipy, sympy

具体怎么用看最下面的样例代码（改也是那里改就好了）就行，关于矩阵性质的具体实现可以看property函数。

##三个功能
1. 解方程
2. 牛顿迭代和二分法
3. 矩阵各种性质

###各种性质如下：

特征值 m.eigen_value
LU分解 m.LU
LDLT分解 m.LDLT
GGT分解 GGT(开始省略'.m')
矩阵自己 matrix
下三角 tril(对角为0)
上三角 triu(对角为0)
对角线 diag
Jaccobi矩阵 Jaccobi
Jaccobi矩阵的谱半径 Jaccobi_converge
Gauss_Seidel矩阵 Gauss_Seidel
Gauss_Seidel矩阵的谱半径 Gauss_Seidel_converge
矩阵求逆 invert
倒置 T
范数1,2,infinite,F (norm1, norm2, norm_infinite, norm_F)
行列式 det
顺序主子式 leading_principle_minor （输出为[D1, D2, D3...]
条件数1,2,infinite (cond1, cond2, cond_infinite)




