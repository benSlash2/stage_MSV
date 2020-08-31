import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def AOB(data, k_s):
    steps = data["steps"]
    ind = data.index
    n = ind.size
    AOB = np.zeros(n)
    for i in ind:
        if steps[i] != 0:
            t = np.arange(n - i)
            AOB[i:] += steps[i] * np.exp(-k_s * 5 * t)
            # for j in range(i, n):
            #     AOB[j] += steps[i]*np.exp(-k_s * 5 * (j-i))
    plt.plot(AOB, 'r--', label='AOB(t)')
    plt.plot(steps, 'b:', label="steps(t)")
    plt.ylabel('values')
    plt.xlabel('time')
    plt.legend(loc='best')
    plt.show()
    return AOB


def R_a(data, i, C_bio, t_max):
    CHO = data["CHO"]
    ind = data.index
    n = ind.size
    Ra = np.zeros(n)
    # for i in ind:
    #     if CHO[i] != 0:
    t = np.arange(n - i)
    Ra[i:] = CHO[i] * C_bio * 5 * t * np.exp(- 5 * t / t_max) / t_max ** 2
    # for j in range(i, n):
    #     Ra[j] = CHO[i] * C_bio * 5 * (j - i) * np.exp(- 5 * (j - i) / t_max) / t_max ** 2
    return Ra


def CPB(data, C_bio, t_max):
    CHO = data["CHO"]
    ind = data.index
    n = ind.size
    CPB = np.zeros(n)
    for i in ind:
        if CHO[i] != 0:
            Ra = R_a(data, i, C_bio, t_max)
            t = np.arange(i, n)
            K = np.array([np.sum(Ra[i:j+1]) for j in t])
            CPB[i:] += CHO[i] * C_bio - 5 * K
            # for j in range(i, n):
            #     K = Ra[range(i, j+1)].sum()
            #     CPB[j] += CHO[i] * C_bio - 5 * K
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

