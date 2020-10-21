import numpy as np
from scipy.integrate import odeint, quad
import matplotlib.pyplot as plt


def aob(data, k_s, plot=False):
    steps = data["steps"]
    ind = data.index
    n = ind.size
    res = np.zeros(n)
    for i in ind:
        if steps[i] != 0 and np.isnan(steps[i]) == 0:
            t = np.arange(n - i)
            res[i:] += steps[i] * np.exp(-k_s * 5 * t)
    if plot:
        plt.plot(res, 'r--', label='AOB(t)')
        plt.plot(steps, 'b:', label="steps(t)")
        plt.ylabel('values')
        plt.xlabel('time')
        plt.legend(loc='best')
        plt.show()
    return res


def cpb(data, c_bio, t_max, plot=False):
    cho = data["CHO"]
    ind = data.index
    n = ind.size
    res = np.zeros(n)
    def ra(x): return quad(lambda t: c_bio * t * np.exp(- t / t_max) / t_max ** 2, 0, 5 * x)[0]
    k = np.array([ra(xi) for xi in ind])
    for i in ind:
        if cho[i] != 0 and np.isnan(cho[i]) == 0:
            res[i:] += res[i] * (c_bio - k[0:n - i])
    if plot:
        plt.plot(res, 'r--', label='CPB(t)')
        plt.plot(cho, 'b:', label='CHO(t)')
        plt.ylabel('values')
        plt.xlabel('time')
        plt.legend(loc='best')
        plt.show()
    return res


def iob(data, k_dia, plot=False):

    def model(z_, _, u_):
        x_ = z_[0]
        y_ = z_[1]
        dx_dt = u_ - k_dia * x_
        dy_dt = k_dia * (x_ - y_)
        dz_dt = [dx_dt, dy_dt]
        return dz_dt

    z0 = [0, 0]
    n = data["datetime"].size
    t = np.linspace(0, (n-1)*5, n)
    u = data["insulin"]
    x = np.empty_like(t)
    y = np.empty_like(t)
    x[0] = z0[0]
    y[0] = z0[1]

    for i in range(1, n):
        t_span = [t[i-1], t[i]]
        z = odeint(model, z0, t_span, args=(u[i],))
        x[i] = z[1][0]
        y[i] = z[1][1]
        z0 = z[1]
    if plot:
        plt.plot(t, u, 'r--', label='u(t)')
        plt.plot(t, x + y, 'b:', label='IOB(t)')
        plt.ylabel('values')
        plt.xlabel('time')
        plt.legend(loc='best')
        plt.show()

    return x + y
