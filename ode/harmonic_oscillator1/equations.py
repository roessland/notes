import numpy as np


class HarmonicOscillator1:
    def __init__(self, k=500.0, m=0.008):
        self.k = k
        self.m = m
        self.name = 'Harmonic oscillator 1'

    def f(self, t, x):
        """
        # f = [fq', fp']'
        # x = [q, p]
        # dq/dt = p
        # dp/dt = -k/m * q
        """

        return np.array([[0.0, 1.0],
                         [-self.k / self.m, 0.0]] @ x)

    def J(self, t, x):
        k = self.k
        m = self.m
        return np.array([[0.0, 1.0],
                         [-self.k / self.m, 0.0]])

    # For symplectic integrators
    def fq(self, t, p):
        return p

    def fp(self, t, q):
        return -self.k / self.m * q


class DoubleSpring1:
    def __init__(self, k0=100.0, m0=1., k1=50.0, m1=1., L0=0.05, L1=0.05):
        self.k0 = k0
        self.m0 = m0
        self.k1 = k1
        self.m1 = m1
        self.L0 = L0
        self.L1 = L1
        self.name = 'Double spring 1'

    def f(self, t, r):
        k0, m0, k1, m1, L0, L1 = self.k0, self.m0, self.k1, self.m1, self.L0, self.L1
        return np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-k0 / m0 - k1 / m0, k1 / m0, 0.0, 0.0],
            [k1 / m1, -k1 / m1, 0.0, 0.0],
        ]) @ r + np.array([
            0,
            0,
            k0 * L0 / m0 - k1 * L1 / m0,
            k1 / m1 * L1,
        ])

    def J(self, t, r):
        k0, m0, k1, m1, L0, L1 = self.k0, self.m0, self.k1, self.m1, self.L0, self.L1
        return np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-k0 / m0 - k1 / m0, k1 / m0, 0.0, 0.0],
            [k1 / m1, -k1 / m1, 0.0, 0.0],
        ])

    def fq(self, t, p):
        return np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ]) @ p + np.array([
            0,
            0,
        ])

    def fp(self, t, q):
        k0, m0, k1, m1, L0, L1 = self.k0, self.m0, self.k1, self.m1, self.L0, self.L1
        return np.array([
            [-k0 / m0 - k1 / m0, k1 / m0],
            [k1 / m1, -k1 / m1],
        ]) @ q + np.array([
            k0 * L0 / m0 - k1 * L1 / m0,
            k1 / m1 * L1,
        ])


class TripleSpring:
    def __init__(self, k0=100.0, m0=1.0, k1=50.0, m1=1.0, k2=75.0, m2=1.0, L0=0.05, L1=0.05, L2=0.05):
        self.k0 = k0
        self.m0 = m0
        self.k1 = k1
        self.m1 = m1
        self.k2 = k2
        self.m2 = m2
        self.L0 = L0
        self.L1 = L1
        self.L2 = L2
        self.name = 'Triple spring'

    def f(self, t, r):
        k0, m0, k1, m1, k2, m2, L0, L1, L2 = self.k0, self.m0, self.k1, self.m1, self.k2, self.m2, self.L0, self.L1, self.L2
        return np.array([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [-k0 / m0 - k1 / m0, k1 / m0, 0.0, 0.0, 0.0, 0.0],
            [k1 / m1, -k1 / m1 - k2 / m1, k2 / m1, 0.0, 0.0, 0.0],
            [0.0, k2 / m2, -k2 / m2, 0.0, 0.0, 0.0]
        ]) @ r + np.array([
            0,
            0,
            0,
            k0 * L0 / m0 - k1 * L1 / m0,
            k1 * L1 / m1 - k2 * L2 / m1,
            k2 * L2 / m2
        ])


class QuadSpring:
    def __init__(self, k0=100.0, m0=1.0, k1=50.0, m1=1.0, k2=75.0, m2=1.0, k3=100.0, m3=0.1, L0=0.05, L1=0.05, L2=0.05,
                 L3=0.05):
        self.k0 = k0
        self.m0 = m0
        self.k1 = k1
        self.m1 = m1
        self.k2 = k2
        self.m2 = m2
        self.k3 = k3
        self.m3 = m3
        self.L0 = L0
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.name = 'Quad spring'

    def f(self, t, r):
        k0, m0, k1, m1, k2, m2, k3, m3, L0, L1, L2, L3 = self.k0, self.m0, self.k1, self.m1, self.k2, self.m2, self.k3, self.m3, self.L0, self.L1, self.L2, self.L3
        return np.array([
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [-k0 / m0 - k1 / m0, k1 / m0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [k1 / m1, -k1 / m1 - k2 / m1, k2 / m1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, k2 / m2, -k2 / m2 - k3 / m2, k3 / m2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, k3 / m3, -k3 / m3, 0.0, 0.0, 0.0, 0.0]
        ]) @ r + np.array([
            0,
            0,
            0,
            0,
            k0 * L0 / m0 - k1 * L1 / m0,
            k1 * L1 / m1 - k2 * L2 / m1,
            k2 * L2 / m2 - k3 * L3 / m2,
            k3 * L3 / m3
        ])

    def J(self, t, r):
        k0, m0, k1, m1, k2, m2, k3, m3, L0, L1, L2, L3 = self.k0, self.m0, self.k1, self.m1, self.k2, self.m2, self.k3, self.m3, self.L0, self.L1, self.L2, self.L3
        return np.array([
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [-k0 / m0 - k1 / m0, k1 / m0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [k1 / m1, -k1 / m1 - k2 / m1, k2 / m1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, k2 / m2, -k2 / m2 - k3 / m2, k3 / m2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, k3 / m3, -k3 / m3, 0.0, 0.0, 0.0, 0.0]
        ])


class QuadSpringZeroed:
    def __init__(self, k0=100.0, m0=1.0, k1=50.0, m1=1.0, k2=75.0, m2=1.0, k3=100.0, m3=0.1):
        self.k0 = k0
        self.m0 = m0
        self.k1 = k1
        self.m1 = m1
        self.k2 = k2
        self.m2 = m2
        self.k3 = k3
        self.m3 = m3
        self.name = 'Quad spring'

    def f(self, t, r):
        k0, m0, k1, m1, k2, m2, k3, m3 = self.k0, self.m0, self.k1, self.m1, self.k2, self.m2, self.k3, self.m3
        return np.array([
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [-k0 / m0 - k1 / m0, k1 / m0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [k1 / m1, -k1 / m1 - k2 / m1, k2 / m1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, k2 / m2, -k2 / m2 - k3 / m2, k3 / m2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, k3 / m3, -k3 / m3, 0.0, 0.0, 0.0, 0.0]
        ]) @ r

    def J(self, t, r):
        k0, m0, k1, m1, k2, m2, k3, m3 = self.k0, self.m0, self.k1, self.m1, self.k2, self.m2, self.k3, self.m3
        return np.array([
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [-k0 / m0 - k1 / m0, k1 / m0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [k1 / m1, -k1 / m1 - k2 / m1, k2 / m1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, k2 / m2, -k2 / m2 - k3 / m2, k3 / m2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, k3 / m3, -k3 / m3, 0.0, 0.0, 0.0, 0.0]
        ])


class DoubleSpring2D:
    """
    Zero spring displacement, only gravity:
    >>> import numpy as np
    >>> eq = DoubleSpring2D(K=np.array([1,1]), L=np.array([1,1]), M=np.array([1,1,1]))
    >>> q = np.array([0.0, 0.0, 0.0, -1.0, 0.0, -2.0])
    >>> p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> r = np.concatenate((q, p))
    >>> eq.f(0.0, r)
    array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
           -9.81,  0.  , -9.81])


    Zero spring displacement, only gravity, sideways initial displacement:
    >>> import numpy as np
    >>> eq = DoubleSpring2D(K=np.array([1,1]), L=np.array([1,1]), M=np.array([1,1,1]))
    >>> q = np.array([0.0, 0.0, -1.0, 0.0, -2.0, 0.0])
    >>> p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> r = np.concatenate((q, p))
    >>> eq.f(0.0, r)
    array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
           -9.81,  0.  , -9.81])

    Displacement for mass 1, vertical displacement:
    >>> import numpy as np
    >>> eq = DoubleSpring2D(K=np.array([1,1]), L=np.array([1,1]), M=np.array([1,1,1]))
    >>> q = np.array([0.0, 0.0, 0.0, -1.1, 0.0, -2.1])
    >>> p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> r = np.concatenate((q, p))
    >>> eq.f(0.0, r)
    array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
           -9.71,  0.  , -9.81])

    Displacement for mass 2, vertical displacement:
    >>> import numpy as np
    >>> eq = DoubleSpring2D(K=np.array([1,1]), L=np.array([1,1]), M=np.array([1,1,1]))
    >>> q = np.array([0.0, 0.0, 0.0, -1.0, 0.0, -2.1])
    >>> p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> r = np.concatenate((q, p))
    >>> eq.f(0.0, r)
    array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
           -9.91,  0.  , -9.71])



    Displacement for mass 2, horizontal displacement:
    >>> import numpy as np
    >>> eq = DoubleSpring2D(K=np.array([1,1]), L=np.array([1,1]), M=np.array([1,1,1]))
    >>> q = np.array([0.0, 0.0, -1.0, 0.0, -2.1, 0.0])
    >>> p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> r = np.concatenate((q, p))
    >>> eq.f(0.0, r)
    array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , -0.1 ,
           -9.81,  0.1 , -9.81])
    """

    def __init__(self, K=None, M=None, L=None):
        if K is None:
            K = np.array([
                100.0, 50.0
            ])
        if M is None:
            M = np.array([
                1.0, 1.0, 1.0
            ])
        if L is None:
            L = np.array([
                1.0, 1.0
            ])

        self.K = K
        self.M = M
        self.L = L
        self.name = 'Double spring 2D'
        self.g = np.array([
            0,
            -9.81,
        ])

    def f(self, t, r):
        assert r.shape[0] == 12
        q = r[0:6]
        p = r[6:]
        return np.concatenate((self.fq(t, p), self.fp(t, q)))

    def fq(self, t, p):
        return p

    def fp(self, t, q):
        K, M, L, g = self.K, self.M, self.L, self.g

        dq_x = (q[2::2] - q[0:-2:2])
        dq_y = (q[3::2] - q[1:-1:2])
        dq = np.sqrt(dq_x ** 2 + dq_y ** 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            e_hat_x = dq_x / dq
            e_hat_y = dq_y / dq
        dL = dq - L
        S = dL * K

        F_prev = np.array([
            0.0, 0.0, -S[0] * e_hat_x[0], -S[0] * e_hat_y[0], -S[1] * e_hat_x[1], -S[1] * e_hat_y[1],
        ])

        F_next = np.array([
            0.0, 0.0, S[1] * e_hat_x[1], S[1] * e_hat_y[1], 0.0, 0.0,
        ])

        G = np.array([
            0.0, 0.0, g[0] * M[1], g[1] * M[1], g[0] * M[2], g[1] * M[2],
        ])

        F = F_prev + F_next + G

        return F / np.tile(M, 2)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
