import numpy as np


def jacobian_num(f, x, eps=1e-10, **params):

    J = np.zeros([len(x), len(x)], dtype=np.float)

    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()

        x1[i] += eps
        x2[i] -= eps

        f1 = f(x1, **params)
        f2 = f(x2, **params)

        J[:, i] = (f1 - f2) / (2 * eps)

    return J