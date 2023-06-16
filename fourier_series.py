import matplotlib.pyplot as plt
from numpy import pi, vectorize, arange, cos, sin, ndarray
from scipy.integrate import quad
from typing import Callable
import matplotlib


'''
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "xelatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

'''


def fourier_series(x_values: ndarray, func: Callable, p: float = pi, limit: int = 30) -> list[float]:
    def cos_term(x: float, n: float) -> Callable:
        return func(x) * cos((n * pi * x) / p)

    def sin_term(x: float, n: float) -> Callable:
        return func(x) * sin((n * pi * x) / p)

    def a0() -> float:
        return 1 / p * quad(func, -p, p)[0]

    def an(n: int) -> float:
        return 1 / p * quad(cos_term, -p, p, args=(n,))[0]

    def bn(n: int) -> float:
        return 1 / p * quad(sin_term, -p, p, args=(n,))[0]

    a0 = a0()
    a = [an(n) for n in range(limit)]
    b = [bn(n) for n in range(limit)]
    fs = [a0 / 2 + sum(a[n] * cos(n * pi * x / p) + b[n] * sin(n * pi * x / p) for n in range(1, limit)) for x in x_values]

    return fs


def g(x: float) -> float:
    if -pi < x <= 0:
        return 0
    elif 0 < x <= pi or x == -pi:
        return cos(x)


def f(x: float) -> float:
    if -pi <= x < 0:
        return -pi
    elif 0 < x <= pi:
        return pi
    return 0


if __name__ == "__main__":
    x = arange(-pi, pi, 0.001)

    y = fourier_series(x, f)
    f = vectorize(f)

    plt.plot(x - 2 * pi, f(x), 'C0')
    plt.plot(x + 2 * pi, f(x), 'C0')
    plt.plot(x, f(x), 'C0', label=r'$f(x)$')
    plt.xticks([-3*pi, -2*pi, -pi, 0, pi, 2 * pi, 3 * pi],
               [r"$-3\pi$", r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$", r"$3\pi$"])
    plt.yticks([-pi, -pi / 2, 0, pi / 2, pi], [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
    plt.legend(loc='upper left')
    plt.axhline(color='black', alpha=0.3)
    plt.axvline(color='black', alpha=0.3)
    plt.show()

    plt.plot(x, y, 'C2', label=r'$S[f](x)$')
    plt.plot(x, f(x), 'C0', label=r'$f(x)$')
    plt.xticks([-pi, -pi / 2, 0, pi / 2, pi], [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
    plt.yticks([-pi, -pi / 2, 0, pi / 2, pi], [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
    plt.legend(loc='upper left')
    plt.axhline(color='black', alpha=0.3)
    plt.axvline(color='black', alpha=0.3)
    plt.show()

    y = fourier_series(x, g)
    g = vectorize(g)

    plt.plot(x - 2 * pi, g(x), 'C0')
    plt.plot(x + 2 * pi, g(x), 'C0')
    plt.plot(x, g(x), 'C0', label=r'$g(x)$')
    plt.xticks([-3 * pi, -2 * pi, -pi, 0, pi, 2 * pi, 3 * pi],
               [r"$-3\pi$", r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$", r"$3\pi$"])
    plt.yticks([-pi, -pi / 2, 0, pi / 2, pi], [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
    plt.legend(loc='upper left')
    plt.axhline(color='black', alpha=0.3)
    plt.axvline(color='black', alpha=0.3)
    plt.show()

    plt.plot(x, y, 'C2', label=r'$S[g](x)$')
    plt.plot(x, g(x), 'C0', label=r'$g(x)$')
    plt.xticks([-pi, -pi / 2, 0, pi / 2, pi], [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
    plt.yticks([-pi, -pi / 2, 0, pi / 2, pi], [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
    plt.legend(loc='upper left')
    plt.axhline(color='black', alpha=0.3)
    plt.axvline(color='black', alpha=0.3)
    plt.show()
