import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag  # type: ignore

from problem_1_5 import TEXT_WIDTH, plot_setup

STEPS = 500

A = np.array([[0.2, 0.6], [0, 0.8]])
B = np.array([[80, 0], [7, 72]])
H = block_diag(A, B)


def func(x):
    return x @ H @ x / 2


def gradient(x):
    return H @ x


def main() -> None:
    eig = np.linalg.eigvals(H)
    lr = 2 / (eig.min() + eig.max())
    xs = [np.array([1, 1, 1, 1])]
    for _ in range(STEPS):
        x = xs[-1]
        xs.append(x - lr * gradient(x))

    eig_1 = np.linalg.eigvals(A)
    eig_2 = np.linalg.eigvals(B)
    lr_1 = 2 / (eig_1.min() + eig_1.max())
    lr_2 = 2 / (eig_2.min() + eig_2.max())
    lr = np.repeat([lr_1, lr_2], 2)
    xs_better = [np.array([1, 1, 1, 1])]
    for _ in range(10):
        x = xs_better[-1]
        xs_better.append(x - lr * gradient(x))

    values = np.array([func(x) for x in xs])
    values_better = np.array([func(x) for x in xs_better])

    plot_setup()
    plt.figure(figsize=(0.6 * TEXT_WIDTH, 0.24 * TEXT_WIDTH))
    plt.plot(values, label='Original', color='black')
    plt.plot(values_better, label='Improved', color='black', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Function value')
    frame = plt.legend(borderpad=0.2).get_frame()
    frame.set_boxstyle('square')  # type: ignore
    frame.set_linewidth(0.5)
    frame.set_edgecolor('black')
    plt.savefig('improved_gd.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
