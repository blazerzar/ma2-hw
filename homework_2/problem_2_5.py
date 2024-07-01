from time import time

import numpy as np


def newton(x_1, gradient, hessian, *, steps=0, timeout=-1):
    end = time() + timeout
    while steps > 0 or time() < end:
        grad = gradient(x_1)
        hess = hessian(x_1)
        x_1 = x_1 - np.linalg.solve(hess, grad)
        steps -= 1
    return x_1


def bfgs(x_1, gradient, *, steps=0, timeout=-1):
    end = time() + timeout
    prev, curr = x_1, x_1
    prev_g, curr_g = gradient(prev), gradient(curr)
    B_k = np.eye(len(x_1))

    while steps > 0 or time() < end:
        # Clip the update step to prevent nan values
        x_k = curr - np.clip(B_k @ curr_g, -1e5, 1e5)
        prev, curr = curr, x_k
        prev_g, curr_g = curr_g, gradient(curr)

        gamma, delta = curr_g - prev_g, curr - prev
        dg_dot = np.dot(delta, gamma)

        if dg_dot > 1e-10:
            V = np.eye(len(x_1)) - np.outer(gamma, delta) / dg_dot
            B_k = V.T @ B_k @ V + np.outer(delta, delta) / dg_dot
        steps -= 1

    return curr
