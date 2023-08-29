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
    def __init__(self, k0=5000.0, m0=0.1, k1=5000.0, m1=0.1, L0=0.1, L1=0.1):
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
            [0.0,            0.0,     1.0,  0.0],
            [0.0,            0.0,     0.0,  1.0],
            [-k0/m0 - k1/m0, k1/m0,   0.0,  0.0],
            [k1/m1,         -k1/m1,   0.0,  0.0],
        ]) @ r + np.array([
            0,
            0,
            k0*L0/m0 - k1*L1/m0,
            k1/m1*L1,
        ])

    def J(self, t, r):
        k0, m0, k1, m1, L0, L1 = self.k0, self.m0, self.k1, self.m1, self.L0, self.L1
        return np.array([
            [0.0,            0.0,     1.0,  0.0],
            [0.0,            0.0,     0.0,  1.0],
            [-k0/m0 - k1/m0, k1/m0,   0.0,  0.0],
            [k1/m1,         -k1/m1,   0.0,  0.0],
        ])

    def fq(self, t, p):
        return np.array([
            [1.0,  0.0],
            [0.0,  1.0],
        ]) @ p + np.array([
            0,
            0,
        ])

    def fp(self, t, q):
        k0, m0, k1, m1, L0, L1 = self.k0, self.m0, self.k1, self.m1, self.L0, self.L1
        return np.array([
            [-k0/m0 - k1/m0,   k1/m0],
            [k1/m1,           -k1/m1],
        ]) @ q + np.array([
            k0*L0/m0 - k1*L1/m0,
            k1/m1*L1,
        ])

