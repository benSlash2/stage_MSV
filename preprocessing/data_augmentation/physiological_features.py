import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def AOB(data, k_s):
    steps = data["steps"]
    ind = data.index
    AOB = np.zeros(ind.size)
    for i in ind:
        if steps[i] != 0:
            for j in range(i, ind[-1] + 1):
                AOB[j] += steps[i]*np.exp(-k_s * 5 * (j-i))
    plt.plot(AOB, 'r--', label='AOB(t)')
    plt.plot(steps, 'b:', label="steps(t)")
    plt.ylabel('values')
    plt.xlabel('time')
    plt.legend(loc='best')
    plt.show()
    return AOB


def R_a (data, C_bio, t_max):
    CHO = data["CHO"]
    ind = data.index
    Ra = np.zeros(ind.size)
    for i in ind:
        if CHO[i] != 0:
            for j in range(i, i + int(t_max/5) + 1):
                Ra[j] += CHO[i] * C_bio * 5 * (j-i) * np.exp(- 5 * (j - i) / t_max) / t_max**2
    return Ra


def CPB(data, C_bio, t_max):
    Ra = R_a(data, C_bio, t_max)
    CHO = data["CHO"]
    ind = data.index
    CPB = np.zeros(ind.size)
    for i in ind:
        if CHO[i] != 0:
            for j in range(i, i + int(t_max/5) + 1):
                K = Ra[range(i,j+1)].sum()
                CPB[j] += CHO[i] * C_bio - 5 * K
    plt.plot(CPB, 'r--', label='CPB(t)')
    plt.plot(CHO, 'b:', label='CHO(t)')
    plt.ylabel('values')
    plt.xlabel('time')
    plt.legend(loc='best')
    plt.show()
    return CPB


def IOB(data, K_DIA):

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

    plt.plot(t, u, 'r--', label='u(t)')
    plt.plot(t, x + y, 'b:', label='IOB(t)')
    plt.ylabel('values')
    plt.xlabel('time')
    plt.legend(loc='best')
    plt.show()

    return x + y

