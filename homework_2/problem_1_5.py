import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STEPS = 10
s = np.array([1, -1]) / np.sqrt(2)
o = np.ones(2) / 4
n = np.array([1, 1])

TEXT_WIDTH = 6.322


def plot_border(ax) -> None:
    """Setup the plot border with a linewidth of 0.5."""
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)


def plot_setup() -> None:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['lines.linewidth'] = 0.6
    plt.rcParams['font.family'] = 'Palatino'
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['axes.labelsize'] = 10


def func(x) -> float:
    """Function f(x, y) = x^2 + e^x + y^2 - xy."""
    x_, y_ = x
    return x_**2 + np.exp(x_) + y_**2 - x_ * y_


def grad(x):
    """Gradient of the function f(x, y) = x^2 + e^x + y^2 - xy."""
    x_, y_ = x
    return np.array([2 * x_ + np.exp(x_) - y_, 2 * y_ - x_])


def f(x):
    """Project to circle x^2 + y^2 = 1.5."""
    if np.dot(x, x) <= 1.5:
        return x
    return np.sqrt(1.5) * x / np.linalg.norm(x)


def g(x):
    """Project to square [-1, 1] x [-1, 1]."""
    return np.clip(x, -1, 1)


def h(x):
    """Projct to triangle with vertices (-1, -1), (1.5, -1), (-1, 1.5)."""
    if np.dot(x - o, n) > 0:
        x = np.dot(x, s) * s + o
    return np.clip(x, -1, 1.5)


def projected_gradient_descent(x_1, gradient, learning_rate, domain, steps):
    """Perform steps of the projected gradient descent algorithm."""
    lr, adaptive, *_ = learning_rate
    xs = [x_1]
    for i in range(steps):
        if adaptive:
            lr = learning_rate[0] / (i + 2)
        x = domain(xs[-1] - lr * gradient(xs[-1]))
        xs.append(x)
    return np.array(xs)


def upper_bound(learning_rate_name, x_1, x_opt, L, alpha, beta, k):
    """Left-hand sides of the inequalities in Theorem 3.3."""
    if learning_rate_name == 'L':
        return L * np.linalg.norm(x_1 - x_opt) / np.sqrt(k + 1)
    if learning_rate_name == 'alpha-beta':
        kappa = beta / alpha
        norm = np.linalg.norm(x_1 - x_opt)
        return beta / 2 * ((kappa - 1) / kappa) ** (2 * k) * norm**2
    if learning_rate_name == 'alpha-L':
        return 2 * L**2 / (alpha * (k + 2))


def lower_bounds(learning_rate_name, xs, x_opt):
    """Right-hand sides of the inequalities in Theorem 3.3."""
    if learning_rate_name == 'L':
        return func(np.mean(xs, axis=0)) - func(x_opt)
    if learning_rate_name == 'alpha-beta':
        return func(xs[-1]) - func(x_opt)
    if learning_rate_name == 'alpha-L':
        T = len(xs)
        x = np.sum(2 * np.arange(1, T + 1).reshape(-1, 1) * xs / (T * (T + 1)), axis=0)
        return func(x) - func(x_opt)


def main() -> None:
    plot_setup()
    x_1 = np.array([-1, 1])
    x_opt = np.array([-0.432563, -0.216281])

    L = np.sqrt(72 + 12 * np.exp(2) + np.exp(4))
    alpha = (4 + np.exp(-2) - np.sqrt(4 + np.exp(-4))) / 2
    beta = (4 + np.exp(2) + np.sqrt(4 + np.exp(4))) / 2

    # Tuples of learning rates and whether the rate is adaptive
    learning_rates = [
        (np.linalg.norm(x_1 - x_opt) / (L * np.sqrt(STEPS + 1)), False, 'L'),
        (1 / beta, False, 'alpha-beta'),
        (2 / alpha, True, 'alpha-L'),
    ]
    domains = [(f, 'circle'), (g, 'square'), (h, 'triangle')]

    print(
        f'Learning rates: L={learning_rates[0][0]:.5f}, '
        f'alpha-beta={learning_rates[1][0]:.5f}, '
        f'alpha-L={learning_rates[2][0] / 2:.5f}-'
        f'{learning_rates[2][0] / (STEPS + 1):.5f} (adaptive)'
    )

    results = pd.DataFrame(
        columns=[
            'domain',
            'learning_rate',
            'x_11',
            'x_opt',
            'lower_bound',
            'upper_bound',
            '||x_11 - x_opt||',
            '|f(x_11) - f(x_opt)|',
        ]
    )
    for domain, d_name in domains:
        _, ax = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 0.3 * TEXT_WIDTH))
        for i, (lr, adaptive, lr_name) in enumerate(learning_rates):
            xs = projected_gradient_descent(x_1, grad, (lr, adaptive), domain, STEPS)
            results.loc[len(results)] = (  # type: ignore
                d_name,
                lr_name,
                xs[-1].round(6),
                x_opt,
                lower_bounds(lr_name, xs, x_opt),
                upper_bound(lr_name, x_1, x_opt, L, alpha, beta, STEPS),
                np.linalg.norm(xs[-1] - x_opt),
                np.abs(func(xs[-1]) - func(x_opt)),
            )

            ax[i].plot(*zip(*xs), label=f'{d_name} {lr_name}', color='black')
            ax[i].plot(*x_1, 'k.')
            ax[i].plot(*x_opt, 'kx')
            ax[i].set_xlim(-1.5, 1.5)
            ax[i].set_ylim(-1.5, 1.5)
            ax[i].set_xlabel('$x_1$')
            ax[i].set_ylabel('$x_2$')
            ax[i].set_aspect('equal')

        plt.savefig(f'{d_name}_convergence.pdf', bbox_inches='tight')
        plt.close()

    print(results)


if __name__ == '__main__':
    main()
