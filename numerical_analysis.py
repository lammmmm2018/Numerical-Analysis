# coding:utf-8
import numpy as np
import scipy.linalg
import math
import sympy
//haha
'''
    @property
    def spectral_radius(self):
        return self._spectral_radius(self.matrix)

    @staticmethod
    def _spectral_radius(item):
        return max(M._eigen_value(item))
'''


class M:
    def __init__(self, raw):
        self.matrix = np.matrix(raw)

    def __repr__(self):
        return str(self.matrix)

    @property
    def eigen_value(self):
        return self._eigen_value(self.matrix)

    @staticmethod
    def _eigen_value(item):
        return np.linalg.eig(item)[0]

    @property
    def LU(self):
        L, U, _ = sympy.Matrix(self.matrix).LUdecomposition()
        return np.matrix(L), np.matrix(U)

    @property
    def LDLT(self):
        L, U = self.LU
        D = np.matrix(np.diagflat(np.diag(U)))
        return L, D, L.T

    @property
    def GGT(self):
        L, D, _ = self.LDLT
        D = D.tolist()
        sqrtD = np.matrix([[math.sqrt(x) for x in D[i]] for i in range(len(D[0]))])
        G = L * sqrtD
        return G, G.T

    @property
    def matrix(self):
        return self.matrix

    @property
    def tril(self):
        return np.tril(self.matrix, k=-1)

    @property
    def triu(self):
        return np.triu(self.matrix, k=1)

    @property
    def diag(self):
        return np.diagflat(np.diag(self.matrix))

    @property
    def Jaccobi_converge(self):
        return M._eigen_value(self.Jaccobi)

    @property
    def Jaccobi(self):
        ja = -1 * np.matmul(np.linalg.inv(self.diag), (self.tril + self.triu))  # -D-1(L+U)
        return ja

    @property
    def Gauss_Seidel(self):
        gs = -1 * np.matmul(np.linalg.inv(self.diag + self.tril), self.triu)
        return gs

    @property
    def Gauss_Seidel_converge(self):
        return M._eigen_value(self.Gauss_Seidel)

    @property
    def invert(self):
        return np.linalg.inv(self.matrix)

    def __mul__(self, other):
        return self.matrix * other.matrix

    def __add__(self, other):
        return self.matrix + other.matrix

    def __sub__(self, other):
        return self.matrix - other.matrix

    def __neg__(self):
        return -self.matrix

    def __getitem__(self, item):
        return self.matrix[item].tolist()[0]

    @property
    def T(self):
        return self.matrix.T

    @property
    def norm1(self):
        return self._norm1(self.matrix)

    @staticmethod
    def _norm1(m):
        return max(map(lambda x: sum(x), abs(m.T).tolist()))

    @property
    def norm2(self):
        return self._norm2(self.matrix)

    @staticmethod
    def _norm2(m):
        return max(map(lambda x: x ** 0.5, M._eigen_value(m.T * m).tolist()))

    @property
    def norm_infinite(self):
        return self._norm_infinite(self.matrix)

    @staticmethod
    def _norm_infinite(m):
        return max(map(lambda x: sum(x), abs(m).tolist()))

    @property
    def norm_F(self):
        return sum(map(lambda x: x ** 2, self.matrix.reshape(1, self.matrix.size).tolist()[0])) ** 0.5

    def __abs__(self):
        return abs(self.matrix)

    def __pow__(self, power, modulo=None):
        return self.matrix ** power

    @property
    def det(self):
        return scipy.linalg.det(self.matrix)

    @property
    def leading_principle_minor(self):
        return self._leading_principle_minor(self.matrix.tolist(), 1, self.matrix.shape[0], [])

    @staticmethod
    def _leading_principle_minor(matrix, k, max_k, result_l):
        tmp_l = [[0 for i in range(k)] for j in range(k)]
        if max_k < k:
            return result_l
        else:
            for i in range(k):
                for j in range(k):
                    tmp_l[i][j] = matrix[i][j]
            result_l.append(M(tmp_l).det)
            return M._leading_principle_minor(matrix, k + 1, max_k, result_l)

    @property
    def cond1(self):
        return self.norm1 * self._norm1(self.matrix.T)

    @property
    def cond2(self):
        return self.norm2 * self._norm2(self.matrix.T)

    @property
    def cond_infinite(self):
        return self.norm_infinite * self._norm_infinite(self.matrix.T)


# m = M('2.0 -1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 -2.0')
# print m.Jaccobi, "\n", m.Jaccobi_converge
# print m.Gauss_Seidel, "\n", m.Gauss_Seidel_converge

def property(m):
    print "--> ", "eigen_value: ", m.eigen_value
    print "--> ", "norm1: ", m.norm1
    print "--> ", "norm_infinite: ", m.norm_infinite
    print "--> ", "norm2: ", m.norm2
    print "--> ", "normF: ", m.norm_F
    print "--> ", "det: ", m.det
    print "--> ", "leading_principle_minor: ", m.leading_principle_minor
    print "--> ", "cond1: ", m.cond1
    print "--> ", "cond2: ", m.cond2
    print "--> ", "cond_infinite: ", m.cond_infinite
    print "--> ", "Jaccobi: \n", m.Jaccobi, '\n'
    print "--> ", "Jaccobi_converge: \n", m.Jaccobi_converge, '\n'
    print "--> ", "Gauss_Seidel: \n", m.Gauss_Seidel, "\n"
    print "--> ", "Gauss_Seidel_converge: \n", m.Gauss_Seidel_converge, "\n"


def solve_equation(A, b):
    if isinstance(b, str):
        b = b.split(' ')
        b = map(int, b)
    A = np.matrix(A)
    b = np.array(b)
    return scipy.linalg.solve(A, b)


def newton(g, x0, stop, k=0):
    x1 = g(x0)
    if abs(x1 - x0) <= stop:
        print k, ' | ', x1, ' | ', x1 - x0
        return x1
    if k == 0:
        print k, ' | ', x0
    else:
        print k, ' | ', x1, ' | ', x1 - x0
    return newton(g, x1, stop, k + 1)


def binary(g, a, b, stop, k=0, old_a=0, old_b=0):
    if k == 0:
        old_a = a
        old_b = b
    x = (a + b) * 1.0 / 2
    gx = g(x)
    flag = '-' if gx < 0 else '+'
    if (old_b - old_a) * 1.0 / (2 ** (k + 1)) <= stop or gx == 0:
        print k, ' | ', a, ' | ', b, ' | ', x, ' | ', flag, ' | ', g(x)
        return g(x)
    print k, ' | ', a, ' | ', b, ' | ', x, ' | ', flag, ' | ', g(x)
    if g(a) < 0:
        if gx < 0:
            a = x
        else:
            b = x
    else:
        if gx < 0:
            b = x
        else:
            a = x
    return binary(g, a, b, stop, k + 1, old_a, old_b)


if __name__ == '__main__':
    #基本操作
    m = M('1 2 3; 2 5 2; 3 1 5')
    property(m)

    #牛顿迭代和二分法
    def g(x):
        return (x + 1) ** (1. / 3)


    def f(x):
        return x ** 3 - x - 1


    newton(g, 1.5, 0.000001) #0.00001为误差限
    binary(f, 1, 1.5, 0.005) 

    #解方程
    print solve_equation('1 2 3; 12 5 6; 7 8 9', '3 2 7')
