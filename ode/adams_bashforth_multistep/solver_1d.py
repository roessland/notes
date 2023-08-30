import numpy as np

from root_finding.newtons_method import newtons_method

lam = 1.1


def f(t, y):
    return lam * y


def dfdx(t, y):
    return lam


def solve_forward_euler(y0, dt, T):
    K = int(T / dt)
    y = np.zeros(K)
    y[0] = y0
    for k in range(K - 1):
        y[k + 1] = y[k] + dt * f(k * dt, y[k])
    return y


def adams_bashforth_2_explicit(y0, dt, T):
    K = int(T / dt)
    y = np.zeros(K)
    y[0] = y0
    y[1] = y[0] + dt * f(0, y[0])  # bootstrap with forward Euler
    for k in range(1, K - 1):
        tj, tk = (k - 1) * dt, k * dt
        yj, yk = y[k - 1], y[k]
        y[k + 1] = y[k] + dt * (3 / 2 * f(tk, yk) - 1 / 2 * f(tj, yj))
    return y


def adams_bashforth_2_implicit(y0, dt, T):
    K = int(T / dt)
    y = np.zeros(K)
    y[0] = y0
    y[1] = y[0] + dt * f(0, y[0])  # bootstrap with forward Euler
    for k in range(1, K - 1):
        tj, tk, tl = (k - 1) * dt, k * dt, (k + 1) * dt
        yj, yk = y[k - 1], y[k]
        fj, fk = f(tj, yj), f(tk, yk)
        y[k + 1] = newtons_method(lambda yl: yl - yk - dt * (-1 / 12 * fj + 2 / 3 * fk + 5 / 12 * f(tl, yl)),
                                  lambda yl: 1 - dt * (5 / 12 * dfdx(tl, yl)), yk)
    return y


def solve_exact(y0, dt, T):
    ts = np.arange(0.0, T, dt)
    y = y0 * np.exp(lam * ts)
    return y


if __name__ == '__main__':
    y0 = 1.0
    dt = 0.1
    T = 10.0
    y = solve_forward_euler(y0, dt, T)
    y_exact = solve_exact(y0, dt, T)
    print(y)
    print(y_exact)
    print(y_exact - y)
