# simple_test.py
import numpy as np
import scipy
import scipy.sparse
import sys
import warnings

# Импортируем наши модули
import optimization
import oracles

def assert_almost_equal(actual, expected, places=7):
    """Аналог assert_almost_equal из nose"""
    if abs(actual - expected) > 10**-places:
        raise AssertionError(f"{actual} != {expected} within {places} places")

def ok_(condition, message=""):
    """Аналог ok_ из nose"""
    if not condition:
        raise AssertionError(message)

def eq_(a, b, message=""):
    """Аналог eq_ из nose"""
    if a != b:
        raise AssertionError(f"{a} != {b}: {message}")

def run_test(test_func, test_name):
    """Запускает тест и обрабатывает исключения"""
    try:
        test_func()
        print(f"✓ {test_name} passed")
        return True
    except Exception as e:
        print(f"✗ {test_name} failed: {e}")
        return False

def test_python3():
    ok_(sys.version_info > (3, 0))

def test_QuadraticOracle():
    # Quadratic function:
    #   f(x) = 1/2 x^T x - [1, 2, 3]^T x
    A = np.eye(3)
    b = np.array([1, 2, 3])
    quadratic = oracles.QuadraticOracle(A, b)

    # Check at point x = [0, 0, 0]
    x = np.zeros(3)
    assert_almost_equal(quadratic.func(x), 0.0)
    ok_(np.allclose(quadratic.grad(x), -b))
    ok_(np.allclose(quadratic.hess(x), A))
    ok_(isinstance(quadratic.grad(x), np.ndarray))
    ok_(isinstance(quadratic.hess(x), np.ndarray))

    # Check at point x = [1, 1, 1]
    x = np.ones(3)
    assert_almost_equal(quadratic.func(x), -4.5)
    ok_(np.allclose(quadratic.grad(x), x - b))
    ok_(np.allclose(quadratic.hess(x), A))

def test_log_reg_basic():
  # Simple data:
    A = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    b = np.array([1, 1, -1, 1])
    reg_coef = 0.5

    # Logistic regression oracle:
    logreg = oracles.create_log_reg_oracle(A, b, reg_coef, oracle_type='usual')

    # Check at point x = [0, 0]
    x = np.zeros(2)
    assert_almost_equal(logreg.func(x), 0.693147180, places=6)
    
    grad = logreg.grad(x)
    ok_(np.allclose(grad, [-0.25, -0.25], atol=1e-6))
    
    #hess = logreg.hess(x)
    #expected_hess = [[0.625, 0.0625], [0.0625, 0.625]]
    #ok_(np.allclose(hess, expected_hess, atol=1e-6))

def test_grad_finite_diff():
    # Quadratic function.
    A = np.eye(3)
    b = np.array([1, 2, 3])
    quadratic = oracles.QuadraticOracle(A, b)
    g = oracles.grad_finite_diff(quadratic.func, np.zeros(3))
    ok_(isinstance(g, np.ndarray))
    ok_(np.allclose(g, -b))

def test_hess_finite_diff():
    # Quadratic function.
    A = np.eye(3)
    b = np.array([1, 2, 3])
    quadratic = oracles.QuadraticOracle(A, b)
    H = oracles.hess_finite_diff(quadratic.func, np.zeros(3))
    ok_(isinstance(H, np.ndarray))
    ok_(np.allclose(H, A))

def test_line_search_constant():
    oracle = oracles.QuadraticOracle(np.eye(3), np.array([1, 2, 3]))
    x = np.array([100, 0, 0])
    d = np.array([-1, 0, 0])

    # Constant line search
    ls_tool = optimization.LineSearchTool(method='Constant', c=1.0)
    assert_almost_equal(ls_tool.line_search(oracle, x, d), 1.0)
    ls_tool = optimization.LineSearchTool(method='Constant', c=10.0)
    assert_almost_equal(ls_tool.line_search(oracle, x, d), 10.0)

def test_gradient_descent_basic():
    class ZeroOracle2D(oracles.BaseSmoothOracle):
        def func(self, x): return 0.0
        def grad(self, x): return np.zeros(2)
        def hess(self, x): return np.zeros([2, 2])

    oracle = ZeroOracle2D()
    x0 = np.ones(2)
    
    # Basic call
    result = optimization.gradient_descent(oracle, x0)
    ok_(len(result) == 3)
    ok_(np.allclose(result[0], x0))
    eq_(result[1], 'success')

def test_newton_basic():
    class ZeroOracle2D(oracles.BaseSmoothOracle):
        def func(self, x): return 0.0
        def grad(self, x): return np.zeros(2)
        def hess(self, x): return np.zeros([2, 2])

    oracle = ZeroOracle2D()
    x0 = np.ones(2)
    
    # Basic call
    result = optimization.newton(oracle, x0)
    ok_(len(result) == 3)
    ok_(np.allclose(result[0], x0))
    eq_(result[1], 'success')

def main():
    print("Running simplified tests...")
    print("=" * 50)
    
    tests = [
        (test_python3, "Python 3 check"),
        (test_QuadraticOracle, "Quadratic Oracle"),
        (test_log_reg_basic, "Logistic Regression Basic"),
        (test_grad_finite_diff, "Gradient Finite Difference"),
        (test_hess_finite_diff, "Hessian Finite Difference"),
        (test_line_search_constant, "Line Search Constant"),
        (test_gradient_descent_basic, "Gradient Descent Basic"),
        (test_newton_basic, "Newton Basic"),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func, test_name in tests:
        if run_test(test_func, test_name):
            passed += 1
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
