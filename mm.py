import matplotlib.pyplot as plt
import matplotlib
from sympy.integrals import integrate
from numpy import pi, linspace, vectorize, arange, nan, cos, sin
from scipy.integrate import quad


def fourier_series(X, f=lambda x: x, L=pi, N=30):
    def fCos(x, n):
        return f(x) * cos((n * pi * x) / L)

    def fSin(x, n):
        return f(x) * sin((n * pi * x) / L)

    def a0():
        return 1 / L * quad(f, -L, L)[0]

    def an(n):
        return 1 / L * quad(fCos, -L, L, args=n)[0]

    def bn(n):
        return 1 / L * quad(fSin, -L, L, args=n)[0]

    a = [an(n) for n in range(N)]
    b = [bn(n) for n in range(N)]
    a0 = a0()

    fs = []
    for x in X:
        fs.append(
            a0 * 0.5 + sum(a[n] * cos(n * pi * x / L) + b[n] * sin(n * pi * x / L) for n in range(1, N)))
    return fs


if __name__ == "__main__":
    """
    x = arange(-pi, pi, 0.001)
    y = vectorize(f)
    plt.plot(x, y(x))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xticks(arange(-pi, pi + pi / 2, step=(pi / 2)), ['-π', '-π/2', '0', 'π/2', 'π'])
    plt.yticks(arange(-pi, pi + pi / 2, step=(pi / 2)), ['-π', '-π/2', '0', 'π/2', 'π'])
    plt.title('Περιττή επέκταση της $f$')
    plt.grid(True)
    plt.show()
    """

    f = lambda x: -pi if -pi <= x < 0 else (pi if (0 < x <= pi) else 0)
    g = lambda x: cos(x) if 0 < x <= pi else 0

    x = arange(-pi, pi, 0.001)
    y = fourier_series(x, f)
    y_ = vectorize(f)

    plt.plot(x - 2 * pi, x, 'C1')
    plt.plot(x + 2 * pi, x, 'C1')
    plt.vlines(-pi, -pi, pi, color='C1', linestyles='dashed')
    plt.vlines(pi, -pi, pi, color='C1', linestyles='dashed')
    plt.vlines(3 * pi, 0, pi, color='C1', linestyles='dashed')
    plt.vlines(-3 * pi, -pi, 0, color='C1', linestyles='dashed')
    plt.plot(x, x, 'C1', label='f(x) = x')
    plt.legend(loc='upper left')
    plt.axhline(color='black', alpha=0.3)
    plt.axvline(color='black', alpha=0.3)
    plt.show()

    plt.vlines(pi, 0, pi, color='C1', linestyles='dashed')
    plt.vlines(-pi, -pi, 0, color='C1', linestyles='dashed')
    plt.plot(x, y, 'C0', label='Fourier Series of f(x) = x')
    plt.plot(x, y_(x), 'C1', label='f(x) = x')
    plt.legend(loc='upper left')
    plt.axhline(color='black', alpha=0.3)
    plt.axvline(color='black', alpha=0.3)
    plt.show()





