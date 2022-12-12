import numpy as np
import typing
from matplotlib import pyplot as plt
import scipy.integrate as integrate


def interpolate(net: np.ndarray, f: np.ndarray) -> typing.Callable:
    a = np.array([np.prod([(x_i - x_j) for j, x_j in enumerate(net) if j != i])
                  for i, x_i in enumerate(net)])

    def polynom(x: float):
        return np.sum(
            [(f_i / a_i) * np.prod([(x - x_j) for j, x_j in enumerate(net) if j != i])
             for i, (f_i, a_i) in enumerate(zip(f, a))]
        )

    return polynom


def solve_gauss(a: np.matrix, f: np.ndarray) -> np.ndarray:
    # Я гарантирую, что a[i, i] > 0

    n = a.shape[0]
    for i in range(n - 1):
        for k in range(i + 1, n):
            f[k] -= f[i] * (a[k, i] / a[i, i])
            for j in range(i + 1, n):
                a[k, j] -= a[i, j] * (a[k, i] / a[i, i])
            a[k, i] = 0
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (f[i] - sum(a[i, k] * x[k] for k in range(i + 1, n))) / a[i, i]
    return x


def approach(net: np.ndarray, f: np.array, n: int):
    A = np.matrix([[nd ** j for j in range(n + 1)] for nd in net])

    alpha_gauss = solve_gauss(A.transpose() * A, (f * A).transpose())
    # print(f"alpha_gauss = {alpha_gauss}")

    def pol(x: float | np.ndarray) -> float | np.ndarray:
        return sum(ai * (x ** i) for i, ai in enumerate(alpha_gauss))

    return pol


def get_st_elm(x_net: np.ndarray, f_net, i, j, method: str, n: int = 0) -> float:
    mask = np.full(x_net.shape, True)
    mask[i] = False
    x_net_i = x_net[mask]
    f_net_i = f_net[mask]
    mask[i] = True
    mask[j] = False
    x_net_j = x_net[mask]
    f_net_j = f_net[mask]
    mask[j] = True

    if method == "interpolate":
        f_i = interpolate(x_net_i, f_net_i)
        f_j = interpolate(x_net_j, f_net_j)
    else:
        f_i = approach(x_net_i, f_net_i, n)
        f_j = approach(x_net_j, f_net_j, n)

    return integrate.quad(lambda x: (f_i(x) - f_j(x)) ** 2, min(x_net), max(x_net))[0]


def main():
    N = 4

    # <<--loading data-->>
    with open('function.txt', 'r') as f:
        points = np.array([[float(x) for x in s.split()] for s in f.readlines()])
    x_net = points[:, 0]
    f_net = points[:, 1]
    print(x_net, f_net)
    # <<!-loading data-!>>

    # <<--building interpolating and approximating functions-->>
    interpol = interpolate(x_net, f_net)
    appr = approach(x_net, f_net, N)
    # <<!-building interpolating and approximating functions-!>>

    # <<--summary graph-->>
    plt.scatter(x_net, f_net, c='k')
    arg_sp = np.linspace(min(x_net), max(x_net), 1_000)
    plt.plot(arg_sp, np.array([interpol(x) for x in arg_sp]), c='b')
    plt.plot(arg_sp, np.array([appr(x) for x in arg_sp]), c='r')
    plt.legend(["actual", "interpolate", "approach"])
    plt.show()
    # <<!-summary graph-!>>

    m_st_interpolate = np.array([[get_st_elm(x_net, f_net, i, j, 'interpolate', N)
                                  for j in range(len(x_net))] for i in range(len(x_net))])

    m_st_approach = np.array([[get_st_elm(x_net, f_net, i, j, 'approach', N)
                               for j in range(len(x_net))] for i in range(len(x_net))])

    print()
    for i in range(len(x_net)):
        for j in range(len(x_net)):
            print(f"{m_st_interpolate[i, j]:09.3f}", end='\t')
        print()

    print()
    for i in range(len(x_net)):
        for j in range(len(x_net)):
            print(f"{m_st_approach[i, j]:09.3f}", end='\t')
        print()

    print(f"{np.mean(m_st_interpolate)=}\n{np.mean(m_st_approach)=}")


if __name__ == '__main__':
    main()
