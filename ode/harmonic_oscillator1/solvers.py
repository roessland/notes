import time
from dataclasses import dataclass

import numpy as np
import scipy.integrate


@dataclass
class SolverResult:
    solver_name: str
    ts: np.array
    xs: np.array
    compute_time: float


class VodeSolver:
    def __init__(self, f, J):
        self.f = f
        self.J = J
        self.name = 'VODE BDF'

    def solve(self, tt, x0):
        compute_time_start = time.process_time()
        dims = x0.shape[0]
        K = tt.shape[0]
        xs = np.zeros((dims, K))
        xs[:, 0] = x0

        r = scipy.integrate.ode(self.f, self.J)
        r = r.set_integrator('vode', method='bdf')
        r = r.set_initial_value(x0, tt[0])

        for k in range(0, K - 1):
            xs[:, k + 1] = r.integrate(tt[k + 1])

        compute_time_end = time.process_time()
        compute_time = compute_time_end - compute_time_start
        return SolverResult(self.name, tt, xs, compute_time)


class ForwardEulerSolver:
    def __init__(self, f):
        self.f = f
        self.name = 'Forward Euler'

    def solve(self, tt, x0):
        compute_time_start = time.process_time()
        f = self.f
        dims = x0.shape[0]
        K = tt.shape[0]
        xs = np.zeros((dims, K))
        xs[:, 0] = x0

        for k in range(0, K - 1):
            tk = tt[k]
            dt = tt[k + 1] - tt[k]
            xs[:, k + 1] = xs[:, k] + dt * f(tk, xs[:, k])

        compute_time_end = time.process_time()
        compute_time = compute_time_end - compute_time_start
        return SolverResult(self.name, tt, xs, compute_time)


class SemiImplicitEulerSolver1:
    def __init__(self, fq, fp, shift=False):
        self.fq = fq
        self.fp = fp
        self.name = 'Semi-implicit Euler (variant 1)'
        self.shift = shift

    def solve(self, tt, x0):
        compute_time_start = time.process_time()
        fq, fp = self.fq, self.fp
        dims = x0.shape[0]
        K = tt.shape[0]
        xs = np.zeros((dims, K))
        xs[:, 0] = x0

        for k in range(0, K - 1):
            tk = tt[k]
            dt = tt[k + 1] - tt[k]

            qn = xs[0, k]
            pn = xs[1, k]
            pl = pn + dt * fp(tk, qn)
            ql = qn + dt * fq(tk, pl)

            xs[0, k + 1] = ql
            xs[1, k + 1] = pl

        compute_time_end = time.process_time()
        compute_time = compute_time_end - compute_time_start
        if self.shift:
            tt = tt[1:]
            xs = xs[:, :-1]
        return SolverResult(self.name, tt, xs, compute_time)


class SemiImplicitEulerSolver2:
    def __init__(self, fq, fp):
        self.fq = fq
        self.fp = fp
        self.name = 'Semi-implicit Euler (variant 2)'

    def solve(self, tt, q0, p0):
        compute_time_start = time.process_time()
        fq, fp = self.fq, self.fp
        q_dims = q0.shape[0]
        p_dims = p0.shape[0]
        K = tt.shape[0]
        qs = np.zeros((q_dims, K))
        ps = np.zeros((p_dims, K))
        qs[:, 0] = q0
        ps[:, 0] = p0

        for k in range(0, K - 1):
            tk = tt[k]
            dt = tt[k + 1] - tt[k]
            qn = qs[:, k] # q = x
            pn = ps[:, k] # p = v
            ql = qn + dt * fq(tk, pn)
            pl = pn + dt * fp(tk, ql)

            qs[:, k + 1] = ql
            ps[:, k + 1] = pl

        compute_time_end = time.process_time()
        compute_time = compute_time_end - compute_time_start
        return SolverResult(self.name, tt, np.vstack([qs, ps]), compute_time)


class SemiImplicitEulerSolverAvg:
    def __init__(self, fq, fp):
        self.fq = fq
        self.fp = fp
        self.name = 'Semi-implicit Euler (avg)'

    def solve(self, tt, x0):
        result1 = SemiImplicitEulerSolver1(self.fq, self.fp).solve(tt, x0)
        result2 = SemiImplicitEulerSolver2(self.fq, self.fp).solve(tt, x0)
        return SolverResult(
            solver_name=self.name,
            ts=tt,
            xs=(result1.xs + result2.xs) / 2,
            compute_time=result1.compute_time + result2.compute_time,
        )


class BackwardEulerSolver:
    def __init__(self, f, J):
        self.f = f
        self.J = J
        self.name = 'Backward Euler'

    def solve(self, tt, x0):
        compute_time_start = time.process_time()
        f, J = self.f, self.J
        dims = x0.shape[0]
        K = tt.shape[0]
        xs = np.zeros((dims, K))
        xs[:, 0] = x0

        for k in range(0, K - 1):
            tk = tt[k]
            dt = tt[k + 1] - tt[k]
            xs[:, k + 1] = np.linalg.solve((np.eye(dims) - dt * J(tk, xs[:, k])), xs[:, k])

        compute_time_end = time.process_time()
        compute_time = compute_time_end - compute_time_start
        return SolverResult(self.name, tt, xs, compute_time)


class VelocityVerletSolver:
    def __init__(self, fq, fp):
        self.name = 'Velocity Verlet'
        self.fq = fq
        self.fp = fp

    # Assumes uniform time steps
    def solve(self, tt, q0, p0):
        compute_time_start = time.process_time()
        q_dims = q0.shape[0]
        p_dims = p0.shape[0]
        K = tt.shape[0]
        qs = np.zeros((q_dims, K))
        ps = np.zeros((p_dims, K))

        qs[:, 0] = q0
        ps[:, 0] = p0

        for k in range(0, K - 1):
            t_k = tt[k]
            t_l = tt[k + 1]
            dt = t_k - t_l # Assuming uniform time steps

            q_k = qs[:, k]
            p_k = ps[:, k]

            # fq_k = dq/dt (t=t_k) = v(t)
            fp_k = self.fp(t_k, q_k) # dv/dt = a

            p_kl = p_k + 0.5 * fp_k * dt # correct
            q_l = q_k + p_kl * dt # correct
            # q_l = q_k + p_k*dt + 0.5 * fp_k * dt^2
            fp_l = self.fp(t_l, q_l) # correct
            p_l = p_kl + 0.5 * dt * fp_l

            qs[:, k + 1] = q_l
            ps[:, k + 1] = p_l

        compute_time_end = time.process_time()
        compute_time = compute_time_end - compute_time_start
        return SolverResult(self.name, tt, np.vstack([qs, ps]), compute_time)
