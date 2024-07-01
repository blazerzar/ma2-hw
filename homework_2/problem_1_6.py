import numpy as np

from problem_1_5 import projected_gradient_descent

a = np.array(
    [
        [3.0, 10, 30],
        [0.1, 10, 35],
        [3.0, 10, 30],
        [0.1, 10, 35],
    ]
)
c = np.array([1.0, 1.2, 3.0, 3.2])
p = np.array(
    [
        [0.36890, 0.1170, 0.2673],
        [0.46990, 0.4387, 0.7470],
        [0.10910, 0.8732, 0.5547],
        [0.03815, 0.5743, 0.8828],
    ]
)


def f(x):
    exponents = -np.sum(a * (x - p) ** 2, axis=1)
    return -np.sum(c * np.exp(exponents))


def grad(x):
    exponents = -np.sum(a * (x - p) ** 2, axis=1)
    exp_grad = -2 * a * (x - p)
    return -np.sum((c * np.exp(exponents)).reshape(-1, 1) * exp_grad, axis=0)


def project(x):
    return np.clip(x, 0, 1)


def find_min(x_1):
    lr = 0
    best_f = float('inf')
    for lr_ in np.logspace(-10, -1, 10):
        xs = projected_gradient_descent(x_1, grad, (lr, False), project, 100000)
        if f(xs[-1]) < best_f:
            lr = lr_
            best_f = f(xs[-1])
    print('Best learning rate:', lr)

    xs = projected_gradient_descent(x_1, grad, (lr, False), project, 100000)
    for steps in np.logspace(1, 5, 5):
        steps = int(steps)
        print(f'After {steps:6} steps:', xs[steps - 1], f(xs[steps - 1]))


def main() -> None:
    x_1 = np.array([0.5, 0.5, 0.5])
    print('Initial point: x₁ =', x_1, 'f(x₁) =', f(x_1))
    find_min(x_1)

    x_1 = np.array([0.1, 0.55, 0.85])
    print('\nInitial point: x₁ =', x_1, 'f(x₁) =', f(x_1))
    find_min(x_1)


if __name__ == '__main__':
    main()
