import numpy as np
import typing
from matplotlib import pyplot as plt


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


def approach(net: np.ndarray, f: np.array):
    N = 9
    A = np.matrix([[nd ** j for j in range(N)] for nd in net])
    alpha = np.linalg.solve(A.transpose() * A, (f * A).transpose())
    np.set_printoptions(precision=4, threshold=5, edgeitems=4, suppress=True)
    alpha = alpha.A1
    print(f"alpha = {alpha}")
    print(f"{np.linalg.cond(A.transpose() * A)=}")
    print(f"{np.linalg.det(A.transpose() * A)=}")
    alpha_gauss = solve_gauss(A.transpose() * A, (f * A).transpose())
    print(f"alpha_gauss = {alpha_gauss}")

    def pol(x: float | np.ndarray) -> float | np.ndarray:
        return sum(ai * (x ** i) for i, ai in enumerate(alpha))

    return pol


def main():
    x = np.linspace(-5, 4, 1_000)
    act = (x ** 3 + 2 * (x ** 2) - 7 * x + 3) + np.random.normal(loc=0.0, scale=5.0, size=x.shape)
    # act = np.sin(2*x) + np.random.normal(loc=0.0, scale=0.075, size=x.shape)
    interpol_node_c = 12
    interpol_net = np.take(x, np.linspace(0, 1_000 - 1, interpol_node_c, dtype=int, endpoint=True))
    interpol_f = np.take(act, np.linspace(0, 1_000 - 1, interpol_node_c, dtype=int, endpoint=True))
    pol = interpolate(interpol_net, interpol_f)

    for i, (node_i, f_i) in enumerate(zip(interpol_net, interpol_f)):
        print(f"{node_i=:.4f}\t{f_i=:.4f}\t{pol(node_i)=:.4f}\tdelta={abs(pol(node_i) - f_i):.4f}")

    approach_node_c = 100
    approach_net = np.take(x, np.linspace(0, 1_000 - 1, approach_node_c, dtype=int, endpoint=True))
    approach_f = np.take(act, np.linspace(0, 1_000 - 1, approach_node_c, dtype=int, endpoint=True))
    apr = approach(approach_net, approach_f)

    sigma_interpol = np.std(act - np.array([pol(x_v) for x_v in x]))
    sigma_approach = np.std(act - apr(x))
    print(f"{sigma_interpol=:.4f}\t{sigma_approach=:.4f}")

    plt.plot(x, act, c='k')
    plt.plot(x, np.array([pol(x_v) for x_v in x]), c='b')
    plt.plot(x, apr(x), c='r')
    # plt.scatter(net, f, c='k', s=4)
    plt.show()


if __name__ == '__main__':
    main()
