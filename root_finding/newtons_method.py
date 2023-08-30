def newtons_method(f, dfdx, x0, tol=1e-6, maxiter=100):
    """
    Solve f(x) = 0 by Newton's method.
    >>> import numpy as np
    >>> f = lambda x: x**2 - 2
    >>> dfdx = lambda x: 2*x
    >>> x0 = 1.0
    >>> x = newton_method(f, dfdx, x0)
    >>> np.abs(x - np.sqrt(2)) < 1e-6
    True
    """
    x = x0
    for i in range(maxiter):
        x = x - f(x) / dfdx(x)
        if abs(f(x)) < tol:
            break
    return x


if __name__ == "__main__":
    import doctest
    doctest.testmod()