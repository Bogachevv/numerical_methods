import numpy as np
import typing
from matplotlib import pyplot as plt


def interpolate(net: np.ndarray, f: np.ndarray) -> typing.Callable:
    def polynom(x: float) -> float:
        s = 0
        for i, xi in enumerate(net):
            c = 1
            for j, xj in enumerate(net):
                if i == j:
                    continue
                c *= (x - xj) / (xi - xj)
            s += c * f[i]
        return s

    return polynom


def approach(net: np.ndarray, f: np.array):
    N = 9
    A = np.matrix([[net[i] ** j for j in reversed(range(N))] for i in range(len(f))])
    alpha = np.linalg.solve(A.transpose() * A, (f * A).transpose())

    def pol(x: float):
        return sum(ai * (x ** i) for i, ai in enumerate(alpha))

    return pol


def main():
    x = np.linspace(-1, 5, 1_000)
    net = np.array([0, 0.465, 1, 4])
    f = np.array([0, 0.879, 0, 0])
    pol = interpolate(net, f)
    y = np.array([pol(xi) for xi in x])
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()
