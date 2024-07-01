import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol, exp, integrate

from utils import plot_border, plots_setup

SAMPLE_SIZE = 10**7
N = 100_000


def f(x):
    """The integrand function."""
    return x ** (-7 / 4) * np.exp(-1 / x)


def estimate_unif(upper_bound, sample_size, lower_bound=1):
    """Estimate the integral using uniform sampling."""
    samples = np.random.uniform(lower_bound, upper_bound, sample_size)
    return np.mean(f(samples)) * (upper_bound - lower_bound)


def q(x, upper_bound):
    """The importance sampling distribution."""
    return 3 / 4 * x ** (-7 / 4) / (1 - upper_bound ** (-3 / 4))


def q_quantile(p, upper_bound):
    """The quantile function of the importance sampling distribution."""
    return (1 - 3 / 4 * p / (3 / 4 / (1 - upper_bound ** (-3 / 4)))) ** (-4 / 3)


def estimate_q(upper_bound, sample_size):
    """Estimate the integral using importance sampling."""
    samples_u = np.random.uniform(0, 1, sample_size)
    samples_q = q_quantile(samples_u, upper_bound)
    return np.mean(f(samples_q) / q(samples_q, upper_bound))


def plot_error():
    upper_bounds = np.arange(10_000, 221_000, 10_000)
    integrals = np.zeros((len(upper_bounds), 2))

    for i, upper_bound in enumerate(upper_bounds):
        unif_estimates = [
            estimate_unif(100_000_000, SAMPLE_SIZE, upper_bound) for _ in range(10)
        ]
        integrals[i] = np.mean(unif_estimates), np.std(unif_estimates)
        print(integrals[i])

    _, ax = plt.subplots(figsize=(4, 2))
    plot_border(ax)
    plt.plot(upper_bounds / 1000, integrals[:, 0], color='black')
    plt.xlim(10, 210)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.yticks(np.linspace(0.2, 1.1, 4) * 1e-3)
    plt.xlabel(r'Upper bound $N$ [thousand]')
    plt.ylabel(r'$\int_N^{10^7} f(x)\ dx$')
    plt.savefig('integral_error.pdf', bbox_inches='tight')


def main() -> None:
    np.random.seed(1)
    plots_setup()

    plot_error()

    x = Symbol('x')
    true_value = integrate(x ** (-7 / 4) * exp(-1 / x), (x, 1, np.inf))

    unif_estimates = [estimate_unif(N, SAMPLE_SIZE) for _ in range(10)]
    q_estimates = [estimate_q(N, SAMPLE_SIZE) for _ in range(10)]

    mean_unif, sd_unif = np.mean(unif_estimates), np.std(unif_estimates)
    mean_q, sd_q = np.mean(q_estimates), np.std(q_estimates)

    print(f'True value:                   {true_value.evalf():.5f}')
    print(f'Uniform estimate:             {mean_unif:.5f} +/- {sd_unif:.5f}')
    print(f'Importance sampling estimate: {mean_q:.5f} +/- {sd_q:.5f}')


if __name__ == '__main__':
    main()
