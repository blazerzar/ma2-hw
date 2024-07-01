import matplotlib.pyplot as plt
import numpy as np
from numpyro.diagnostics import autocovariance, effective_sample_size  # type: ignore
from scipy.stats import multivariate_normal  # type: ignore

from utils import plot_border, plots_setup

data_samples = np.array([0.5, 0.4, 0.6, 0.8, 0.3])


def prior(alpha: float, eta: float) -> float:
    """The prior distribution of the parameters."""
    return np.exp(-alpha - 2 * eta) * eta


def weibull(x, alpha: float, eta: float):
    """The Weibull distribution."""
    return alpha * eta * x ** (alpha - 1) * np.exp(-(x**alpha) * eta)


def posterior(alpha: float, eta: float) -> float:
    """The posterior distribution of the parameters."""
    if alpha <= 0 or eta <= 0:
        return 0
    return np.prod(weibull(data_samples, alpha, eta)) * prior(alpha, eta)


def normal_proposal(cov):
    """A normal proposal distribution that generates random samples from a
    multivariate normal distribution with the given covariance matrix.
    """

    def f(alpha: float, eta: float) -> tuple[float, float]:
        alpha_prime, eta_prime = np.random.multivariate_normal([0, 0], cov)
        return alpha_prime, eta_prime

    return f


def exponential_proposal(alpha: float, eta: float) -> tuple[float, float]:
    """An exponential proposal distribution that generates random samples from
    two exponential distributions with the given rate parameters.
    """
    return np.random.exponential(alpha), np.random.exponential(eta)


def normal_transition(cov):
    """The normal probability of transitioning to the new state."""

    def f(alpha: float, eta: float, alpha_prime: float, eta_prime: float) -> float:
        return multivariate_normal.pdf([alpha_prime, eta_prime], cov=cov)

    return f


def exponential_transition(
    alpha: float, eta: float, alpha_prime: float, eta_prime: float
) -> float:
    """The exponential probability of transitioning to the new state."""
    return np.exp(-alpha_prime / alpha) * np.exp(-eta_prime / eta) / (alpha * eta)


def metropolis_hastings(proposal, transition, initial_state, steps):
    """Generate samples from a Markov chain using the Metropolis-Hastings
    algorithm. We want to obtain samples from the posterior distribution.
    The proposal function generates a new state given the current state.
    New state is accepted taking into account the posterior and transition
    probabilities.
    """
    previous_sample = initial_state
    samples = np.zeros((steps, 2))

    for i in range(steps):
        alpha_prime, eta_prime = proposal(*previous_sample)

        rejection = (
            posterior(alpha_prime, eta_prime)
            * transition(alpha_prime, eta_prime, *previous_sample)
        ) / (
            posterior(*previous_sample)
            * transition(*previous_sample, alpha_prime, eta_prime)
        )

        if np.random.uniform() <= rejection:
            previous_sample = alpha_prime, eta_prime
        samples[i] = previous_sample

    return samples


def diagnostics(chains, name):
    """Print the diagnostics of the Markov chain including the mean, variance
    and the effective sample size of the parameters. Also, plot the trace,
    autocovariance and the scatter plot of the chains. We also estimate the
    probability of the parameters being greater than or equal to 2.
    """
    print('P(α >= 2, η >= 2) =', np.mean((chains >= 2).all(axis=2)))

    # 2D scatter plot
    plt.figure(figsize=(3, 3))
    plt.scatter(
        chains[:, :, 0],
        chains[:, :, 1],
        color='black',
        alpha=0.1,
        s=32,
        edgecolors='none',
    )
    plt.gca().set_aspect('equal')
    plt.xlim(0, 6)
    plt.ylim(0, 6)
    plt.xticks([0, 2, 4, 6])
    plt.yticks([0, 2, 4, 6])
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\eta$')
    plot_border(plt.gca())
    plt.savefig(f'{name}_scatter.pdf', bbox_inches='tight')
    plt.close()

    # Diagnostics and plots per chain
    _, ax = plt.subplots(2, 2, figsize=(8, 4), width_ratios=[1.6, 1])
    for i, a in enumerate(ax.flatten()):
        plot_border(a)
        if i % 2:
            # Right column
            a.set_xlim(0, 50)
            a.set_ylabel(r'Autocovariance')
        else:
            # Left column
            a.set_xlim(0, 1000)
            a.set_ylabel(r'$\alpha$' if i == 0 else r'$\eta$')

        if i >= 2:
            a.set_xlabel('Iteration' if i == 2 else r'Lag-$k$')

    for i, chain in enumerate(chains):
        # Diagnostic statistics
        mean_alpha, mean_eta = np.mean(chain, axis=0)
        var_alpha, var_eta = np.var(chain, axis=0)
        ess_alpha = effective_sample_size(chain[:, [0]].T)
        ess_eta = effective_sample_size(chain[:, [1]].T)
        print(
            f'E[α] = {mean_alpha:.3f}, Var[α] = {var_alpha:.3f}, '
            f'ESS = {ess_alpha:.0f}\t'
            f'E[η] = {mean_eta:.3f}, Var[η] = {var_eta:.3f}, '
            f'ESS = {ess_eta:.0f}'
        )

        # Traces
        ax[0, 0].plot(chain[:, 0], alpha=0.8)
        ax[1, 0].plot(chain[:, 1], alpha=0.8)

        # Lag-k autocovariances
        chain_cov = autocovariance(chain)
        ax[0, 1].plot(chain_cov[:52, 0], alpha=0.8)
        ax[1, 1].plot(chain_cov[:52, 1], alpha=0.8)
    print()

    plt.savefig(f'{name}_diagnostics.pdf', bbox_inches='tight')
    plt.close()


def main() -> None:
    np.random.seed(1)
    plots_setup()

    cov = np.array([[4, 1], [1, 4]])
    initial_state = 1, 1
    chain_length = 1000
    num_chains = 5

    for p, t, name in zip(
        (normal_proposal(cov), exponential_proposal),
        (normal_transition(cov), exponential_transition),
        ('normal', 'exponential'),
    ):
        chains = np.array(
            [
                metropolis_hastings(p, t, initial_state, chain_length)
                for _ in range(num_chains)
            ]
        )
        diagnostics(chains, name)

    # Probability estimation
    chain_mvn = metropolis_hastings(
        normal_proposal(cov), normal_transition(cov), initial_state, 100000
    )[5000:]
    chain_exp = metropolis_hastings(
        exponential_proposal, exponential_transition, initial_state, 100000
    )[5000:]
    print(f'P(α >= 2, η >= 2) = {np.mean((chain_mvn >= 2).all(axis=1)):.4f}')
    print(f'P(α >= 2, η >= 2) = {np.mean((chain_exp >= 2).all(axis=1)):.4f}')


if __name__ == '__main__':
    main()
