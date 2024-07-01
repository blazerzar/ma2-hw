import numpy as np

from problem_2_4 import gradient_descent
from problem_2_5 import bfgs, newton


def l_bfgs(x_1, gradient, *, steps=0, m=20):
    prev, curr = x_1, x_1 - 1e-5 * gradient(x_1)
    prev_g, curr_g = gradient(prev), gradient(curr)
    gammas, deltas = [curr_g - prev_g], [curr - prev]

    for _ in range(steps - 1):
        # Compute r = B_k @ curr_g
        q = gradient(curr)

        alphas = []
        for gamma, delta in zip(gammas[::-1], deltas[::-1]):
            dg_dot = np.dot(delta, gamma)
            if dg_dot < 1e-16:
                continue
            alphas.append(np.dot(delta, q) / dg_dot)
            q = q - alphas[-1] * gamma

        # Get initial B_k @ curr_g approximation
        gk, dk = gammas[-1], deltas[-1]
        gk_dot = np.dot(gk, gk)
        r = q if gk_dot < 1e-16 else np.dot(dk, gk) / gk_dot * q

        for gamma, delta, alpha in zip(gammas, deltas, alphas[::-1]):
            dg_dot = np.dot(delta, gamma)
            if dg_dot < 1e-16:
                continue
            beta = np.dot(gamma, r) / dg_dot
            r = r + (alpha - beta) * delta

        prev, curr = curr, curr - r
        prev_g, curr_g = curr_g, gradient(curr)

        # Limit history size
        gammas.append(curr_g - prev_g)
        deltas.append(curr - prev)
        if len(gammas) > m:
            gammas.pop(0)
            deltas.pop(0)

    return curr


def fit_lr(method, gradient, hessian_lr, name, steps):
    # Parameter hessian_lr contains the learning rate or the Hessian matrix
    x_1 = np.array([0.1, 0.1])
    args = (x_1, gradient, hessian_lr) if hessian_lr else (x_1, gradient)
    k, n = method(*args, steps=steps)
    print(f'{name:>6}: k={k:.2f}, n={n:.2f}')


def main() -> None:
    np.random.seed(1)

    for i, N in enumerate((50, 100, 1_000, 10_000, 100_000, 1_000_000)):
        print('\n' if i > 0 else '', end='')
        print(f'N={N}:')
        x = np.arange(1, N + 1)
        y = x + np.random.uniform(0, 1, N)
        X = np.vstack([x, np.ones_like(x)]).T

        def grad(parameters):
            return np.dot(X @ parameters - y, X) / N

        def grad_stochastic(parameters):
            i = np.random.randint(N)
            x = X[i]
            return (np.dot(parameters, x) - y[i]) * x

        def hessian(_):
            return X.T @ X / N

        lr = [1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12][i]
        steps = [100000, 100000, 1000, 1000, 1000, 1000][i]
        fit_lr(gradient_descent, grad, lr, 'GD', steps=steps)
        fit_lr(gradient_descent, grad_stochastic, lr, 'SGD', steps=steps)
        fit_lr(newton, grad, hessian, 'Newton', steps=1000)
        fit_lr(bfgs, grad, None, 'BFGS', steps=1000)
        fit_lr(l_bfgs, grad, None, 'L-BFGS', steps=1000)


if __name__ == '__main__':
    main()
