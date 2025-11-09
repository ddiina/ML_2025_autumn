import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict

class LineSearchTool(object):
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        if self._method == 'Constant':
            return self.c
        elif self._method == 'Armijo':
            alpha = previous_alpha if previous_alpha is not None else self.alpha_0
            c1 = self.c1
            phi0 = oracle.func(x_k)
            phi_prime0 = oracle.grad_directional(x_k, d_k, 0)
            while True:
                phi_alpha = oracle.func_directional(x_k, d_k, alpha)
                if phi_alpha <= phi0 + c1 * alpha * phi_prime0:
                    return alpha
                alpha *= 0.5
                if alpha < 1e-8:
                    return None
        elif self._method == 'Wolfe':
            alpha = previous_alpha if previous_alpha is not None else self.alpha_0
            c1, c2 = self.c1, self.c2
            phi0 = oracle.func(x_k)
            phi_prime0 = oracle.grad_directional(x_k, d_k, 0)
            alpha_prev = 0
            alpha_curr = alpha
            for _ in range(50):
                phi_alpha = oracle.func_directional(x_k, d_k, alpha_curr)
                if phi_alpha > phi0 + c1 * alpha_curr * phi_prime0:
                    # Backtracking
                    alpha_curr *= 0.5
                else:
                    # Check Wolfe condition
                    phi_prime_alpha = oracle.grad_directional(x_k, d_k, alpha_curr)
                    if phi_prime_alpha < c2 * phi_prime0:
                        alpha_curr *= 1.1
                    else:
                        return alpha_curr
            return None
        else:
            return None

def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()

def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    for iteration in range(max_iter):
        g = oracle.grad(x_k)
        g_norm = np.linalg.norm(g)
        if trace:
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(g_norm)
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))
        if g_norm < tolerance:
            return x_k, 'success', history
        d_k = -g
        alpha = line_search_tool.line_search(oracle, x_k, d_k)
        if alpha is None:
            return x_k, 'linesearch_failed', history
        x_k = x_k + alpha * d_k
        if display:
            print(f"Iter {iteration}, func={oracle.func(x_k):.6f}, grad_norm={g_norm:.6f}, alpha={alpha}")
    return x_k, 'iterations_exceeded', history

def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    for iteration in range(max_iter):
        g = oracle.grad(x_k)
        g_norm = np.linalg.norm(g)
        if trace:
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(g_norm)
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))
        if g_norm < tolerance:
            return x_k, 'success', history
        H = oracle.hess(x_k)
        # Проверим, что гессиан положительно определен, используем Холецкого
        try:
            c, lower = scipy.linalg.cho_factor(H)
            p = scipy.linalg.cho_solve((c, lower), -g)
        except LinAlgError:
            return x_k, 'newton_direction_error', history
        alpha = line_search_tool.line_search(oracle, x_k, p)
        if alpha is None:
            return x_k, 'linesearch_failed', history
        x_k = x_k + alpha * p
        if display:
            print(f"Iter {iteration}, func={oracle.func(x_k):.6f}, grad_norm={g_norm:.6f}, alpha={alpha}")
    return x_k, 'iterations_exceeded', history