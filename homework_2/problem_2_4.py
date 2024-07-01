from time import time

import numpy as np


def gradient_descent(x_1, gradient, learning_rate, *, steps=0, timeout=-1):
    end = time() + timeout
    while time() < end or steps > 0:
        x_1 = x_1 - learning_rate * gradient(x_1)
        steps -= 1
    return x_1


def polyak_gd(x_1, gradient, learning_rate, momentum, *, steps=0, timeout=-1):
    end = time() + timeout
    prev, curr = x_1, x_1
    while time() < end or steps > 0:
        x_k = curr - learning_rate * gradient(curr) + momentum * (curr - prev)
        prev, curr = curr, x_k
        steps -= 1
    return curr


def nesterov_gd(x_1, gradient, learning_rate, momentum, *, steps=0, timeout=-1):
    end = time() + timeout
    prev, curr = x_1, x_1
    while steps > 0 or time() < end:
        offset = momentum * (curr - prev)
        x_k = curr - learning_rate * gradient(curr + offset) + offset
        prev, curr = curr, x_k
        steps -= 1
    return curr


def ada_grad(x_1, gradient, learning_rate, *, steps=0, timeout=-1):
    end = time() + timeout
    gradients_sum = np.zeros_like(x_1) + 1e-8
    while time() < end or steps > 0:
        grad = gradient(x_1)
        gradients_sum = gradients_sum + grad**2
        D_k = np.diag(np.power(gradients_sum, -0.5))
        x_1 = x_1 - learning_rate * D_k @ grad
        steps -= 1
    return x_1
