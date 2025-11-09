import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = (1/m) sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = np.asarray(b)
        self.regcoef = float(regcoef)
        self.m = b.shape[0]

    def func(self, x):
        # Вычисляем A x
        Ax = self.matvec_Ax(x)
        # z = -b * (A x)
        z = -self.b * Ax
        # Используем logaddexp для численной стабильности
        # log(1 + exp(z))
        log_terms = np.logaddexp(0, z)
        return np.sum(log_terms) / self.m + (self.regcoef / 2) * np.dot(x, x)

    def grad(self, x):
      Ax = self.matvec_Ax(x)
      p = expit(self.b * Ax)  # sigma(b * A*x)
      # Тогда sigma(-b * A*x) = 1 - p
      grad_vec = -(1 - p) * self.b  # = -b * (1 - p) = -b * sigma(-b * A*x)
      grad_loss = self.matvec_ATx(grad_vec) / self.m + self.regcoef * x
      return grad_loss

    def hess(self, x):
        Ax = self.matvec_Ax(x)
        p = expit(-self.b * Ax)
        W = p * (1.0 - p)
        H = self.matmat_ATsA(W) / float(self.m)
        H = np.asarray(H, dtype=float)
        H[np.diag_indices_from(H)] += self.regcoef
        return H
        #return self.matmat_ATsA(W) / self.m + self.regcoef * np.eye(len(x))

class LogRegL2OptimizedOracle(LogRegL2Oracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self._Ax = None  # Для хранения
        self._Ad = None

    def func_directional(self, x, d, alpha):
        Ax = self.matvec_Ax(x)
        Ad = self.matvec_Ax(d)
        self._Ax = Ax
        self._Ad = Ad
        Ax_new = Ax + alpha * Ad
        z = -self.b * Ax_new
        return np.sum(np.logaddexp(0, z)) / self.m + (self.regcoef / 2) * np.dot(x + alpha * d, x + alpha * d)

    def grad_directional(self, x, d, alpha):
        if self._Ax is None or self._Ad is None:
            self._Ax = self.matvec_Ax(x)
            self._Ad = self.matvec_Ax(d)
        Ax_new = self._Ax + alpha * self._Ad
        p = expit(self.b * Ax_new)
        residual = p - np.ones_like(self.b)
        grad_loss = self.matvec_ATx(residual) / self.m + self.regcoef * (x + alpha * d)
        return grad_loss.dot(d)
    
def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Создает оракул логистической регрессии.
    """
    # Для плотных матриц:
    if scipy.sparse.isspmatrix(A):
        matvec_Ax = lambda x: A @ x
        matvec_ATx = lambda x: A.T @ x
        def matmat_ATsA(s):
            # A.T @ diag(s) @ A
            # Для sparse: scipy.sparse.diags(s)
            return A.T @ (scipy.sparse.diags(s) @ A)
    else:
        # Для dense
        matvec_Ax = lambda x: A @ x
        matvec_ATx = lambda x: A.T @ x
        def matmat_ATsA(s):
            return A.T @ (s * A)

    if oracle_type == 'usual':
        oracle_class = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle_class = LogRegL2OptimizedOracle
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)

    return oracle_class(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

# Реализация финитных разностных приближений
def grad_finite_diff(func, x, eps=1e-8):
    """
    Приближенный градиент через конечные разности.
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = 1.0
        grad[i] = (func(x + eps * e) - func(x)) / eps
    return grad

def hess_finite_diff(func, x, eps=1e-5):
    """
    Приближенный гессиан через конечные разности.
    """
    n = len(x)
    hessian = np.zeros((n, n))
    f_x = func(x)
    for i in range(n):
        e_i = np.zeros_like(x)
        e_i[i] = 1.0
        f_x_plus = func(x + eps * e_i)
        for j in range(i, n):
            e_j = np.zeros_like(x)
            e_j[j] = 1.0
            f_x_plus_ej = func(x + eps * e_j)
            f_x_plus_ei_ej = func(x + eps * (e_i + e_j))
            hessian[i, j] = (f_x_plus_ei_ej - f_x_plus - f_x_plus_ej + f_x) / (eps ** 2)
            if i != j:
                hessian[j, i] = hessian[i, j]
    return hessian
