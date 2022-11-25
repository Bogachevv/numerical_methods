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


def approach(net: np.ndarray, f: np.array, n: int):
    A = np.matrix([[nd ** j for j in range(n + 1)] for nd in net])

    # DEBUG: delete in release version
    alpha = np.linalg.solve(A.transpose() * A, (f * A).transpose())
    np.set_printoptions(precision=4, threshold=5, edgeitems=4, suppress=True)
    alpha = alpha.A1
    print(f"alpha = {alpha}")
    print(f"{np.linalg.cond(A.transpose() * A)=}")
    print(f"{np.linalg.det(A.transpose() * A)=}")
    # !DEBUG

    alpha_gauss = solve_gauss(A.transpose() * A, (f * A).transpose())
    print(f"alpha_gauss = {alpha_gauss}")

    def pol(x: float | np.ndarray) -> float | np.ndarray:
        return sum(ai * (x ** i) for i, ai in enumerate(alpha))

    return pol


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


if __name__ == '__main__':
    main()
