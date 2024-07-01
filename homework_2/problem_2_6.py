import numpy as np
from sympy import diff, lambdify, symbols

from problem_2_4 import ada_grad, gradient_descent, nesterov_gd, polyak_gd
from problem_2_5 import bfgs, newton


def f_a(x_):
    x, y, z = x_
    return (x - z) ** 2 + (2 * y + z) ** 2 + (4 * x - 2 * y + z) ** 2 + x + y


grad_a_matrix = np.array([[34, -16, 6, 1], [-16, 16, 0, 1], [6, 0, 6, 0]])


def grad_a(x):
    return grad_a_matrix @ np.concatenate([x, [1]])


def hessian_a(_):
    return np.array([[34, -16, 6], [-16, 16, 0], [6, 0, 6]])


def f_b(x_):
    x, y, z = x_
    return (x - 1) ** 2 + (y - 1) ** 2 + 100 * (y - x**2) ** 2 + 100 * (z - y**2) ** 2


def grad_b(x_):
    x, y, z = x_
    a, b = y - x**2, z - y**2
    return np.array(
        [
            2 * (x - 1) - 400 * x * a,
            2 * (y - 1) + 200 * a - 400 * y * b,
            200 * b,
        ]
    )


def hessian_b(x_):
    x, y, z = x_
    return np.array(
        [
            [2 - 400 * y + 1200 * x**2, -400 * x, 0],
            [-400 * x, 202 - 400 * z + 1200 * y**2, -400 * y],
            [0, -400 * y, 200],
        ]
    )


def f_c(x_):
    x, y = x_
    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y**2) ** 2
        + (2.625 - x + x * y**3) ** 2
    )


def grad_c(x_):
    x, y = x_
    a, b, c = 1.5 - x + x * y, 2.25 - x + x * y**2, 2.625 - x + x * y**3
    return np.array(
        [
            2 * (y - 1) * a + 2 * (y**2 - 1) * b + 2 * (y**3 - 1) * c,
            2 * x * a + 4 * x * y * b + 6 * x * y**2 * c,
        ]
    )


def evaluate(func, grad, hessian, initial_point, *, steps=0, timeout=-1):
    x_1 = np.array(initial_point)

    # Gradient descent
    results_gd = [
        (gradient_descent(x_1, grad, lr, steps=steps, timeout=timeout), lr)
        for lr in (1e-6, 1e-5, 1e-4, 1e-2)
    ]
    best_gd = np.nanargmin([func(r) for r, _ in results_gd])
    x_k, lr = results_gd[best_gd]
    print_results('GD', initial_point, func(x_k), x_k, lr)

    # Polyak
    results_polyak = [
        (polyak_gd(x_1, grad, lr, 0.1, steps=steps, timeout=timeout), lr)
        for lr in (1e-5, 1e-4, 1e-2)
    ]
    best_polyak = np.nanargmin([func(r) for r, _ in results_polyak])
    x_k, lr = results_polyak[best_polyak]
    print_results('Polyak', initial_point, func(x_k), x_k, lr)

    # Nesterov
    results_nesterov = [
        (nesterov_gd(x_1, grad, lr, 0.1, steps=steps, timeout=timeout), lr)
        for lr in (1e-5, 1e-4, 1e-2)
    ]
    best_nesterov = np.nanargmin([func(r) for r, _ in results_nesterov])
    x_k, lr = results_nesterov[best_nesterov]
    print_results('Nesterov', initial_point, func(x_k), x_k, lr)

    # AdaGrad
    results_ada = [
        (ada_grad(x_1, grad, lr, steps=steps, timeout=timeout), lr)
        for lr in (1e-5, 1e-4, 1e-2)
    ]
    best_ada = np.nanargmin([func(r) for r, _ in results_ada])
    x_k, lr = results_ada[best_ada]
    print_results('AdaGrad', initial_point, func(x_k), x_k, lr)

    # Newton method
    x_k = newton(x_1, grad, hessian, steps=steps, timeout=timeout)
    print_results('Newton', initial_point, func(x_k), x_k)

    # BFGS
    x_k = bfgs(x_1, grad, steps=steps, timeout=timeout)
    print_results('BFGS', initial_point, func(x_k), x_k)


def print_results(method, x_1, value, x_k, lr=0):
    lr_str = f'lr={lr:4.0e},' if lr else ' ' * 9
    x_1_str = f'{str(x_1):>15}'
    print(f'  {method:>15} {x_1_str}: {lr_str} {value:15.5f} at {x_k.round(3)}')


def evaluate_function(func, grad, hessian, initial_points):
    for steps in (2, 5, 10, 100):
        print(f'Steps={steps}:')
        evaluate(func, grad, hessian, initial_points[0], steps=steps)
        evaluate(func, grad, hessian, initial_points[1], steps=steps)
    for timeout in (0.1, 1, 2):
        print(f'Timeout={timeout}:')
        evaluate(func, grad, hessian, initial_points[0], timeout=timeout)
        evaluate(func, grad, hessian, initial_points[1], timeout=timeout)


def main() -> None:
    np.seterr(all='ignore')

    # Compute the last Hessian symbolically
    x, y = symbols('x y')
    f = (
        (1.5 - x - x * y) ** 2
        + (2.25 - x + x * y**2) ** 2
        + (2.625 - x + x * y**3) ** 2
    )
    f_xx = lambdify((x, y), diff(diff(f, x), x))
    f_xy = lambdify((x, y), diff(diff(f, x), y))
    f_yy = lambdify((x, y), diff(diff(f, y), y))

    def hessian_c(x):
        return np.array([[f_xx(*x), f_xy(*x)], [f_xy(*x), f_yy(*x)]])

    print('Function A:')
    evaluate_function(f_a, grad_a, hessian_a, [(0, 0, 0), (1, 1, 0)])

    print('\nFunction B:')
    evaluate_function(f_b, grad_b, hessian_b, [(1.2, 1.2, 1.2), (-1, 1.2, 1.2)])

    print('\nFunction C:')
    evaluate_function(f_c, grad_c, hessian_c, [(1, 1), (4.5, 4.5)])


if __name__ == '__main__':
    main()
