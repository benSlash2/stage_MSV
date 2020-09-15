import numpy as np
from scipy.integrate import odeint, quad
import matplotlib.pyplot as plt


def AOB(data, k_s, plot=False):
    steps = data["steps"]
    ind = data.index
    n = ind.size
    AOB = np.zeros(n)
    for i in ind:
        if steps[i] != 0 and np.isnan(steps[i]) == 0:
            t = np.arange(n - i)
            AOB[i:] += steps[i] * np.exp(-k_s * 5 * t)
    if plot:
        plt.plot(AOB, 'r--', label='AOB(t)')
        plt.plot(steps, 'b:', label="steps(t)")
        plt.ylabel('values')
        plt.xlabel('time')
        plt.legend(loc='best')
        plt.show()
    return AOB


def CPB(data, C_bio, t_max, plot=False):
    CHO = data["CHO"]
    ind = data.index
    n = ind.size
    CPB = np.zeros(n)
    Ra = lambda x: quad(lambda t: C_bio * t * np.exp(- t / t_max) / t_max ** 2, 0, 5 * x)[0]
    K = np.array([Ra(xi) for xi in ind])
    for i in ind:
        if CHO[i] != 0 and np.isnan(CHO[i]) == 0:
            CPB[i:] += CHO[i] * (C_bio - K[0:n-i])
    if plot:
        plt.plot(CPB, 'r--', label='CPB(t)')
        plt.plot(CHO, 'b:', label='CHO(t)')
        plt.ylabel('values')
        plt.xlabel('time')
        plt.legend(loc='best')
        plt.show()
    return CPB


def IOB(data, K_DIA, plot=False):

    def model(z,t,u):
        x = z[0]
        y = z[1]
        dxdt = u - K_DIA * x
        dydt = K_DIA * (x - y)
        dzdt = [dxdt, dydt]
        return dzdt

    z0 = [0, 0]
    n = data["datetime"].size
    t = np.linspace(0, (n-1)*5, n)
    u = data["insulin"]
    x = np.empty_like(t)
    y = np.empty_like(t)
    x[0] = z0[0]
    y[0] = z0[1]

    for i in range(1, n):
        tspan = [t[i-1], t[i]]
        z = odeint(model, z0, tspan, args=(u[i],))
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

