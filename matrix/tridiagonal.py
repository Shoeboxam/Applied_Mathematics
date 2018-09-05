import numpy as np
from scipy import sparse

np.set_printoptions(suppress=True, precision=3)


def tridiagonal(vals, N):
    return sparse.diags(vals, [-1, 0, 1], shape=(N, N)).toarray()


A = tridiagonal(np.random.uniform(-1, 1, size=3), 5)
print(A)


# determinants of tridiagonal matrices may be constructed quickly
def cheap_det(A):
    cache = {}

    def det(A):
        if A.shape[0] in cache:
            return cache[A.shape[0]]

        if np.shape(A) == (1, 1):
            return A[0, 0]
        if np.shape(A) == (2, 2):
            return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

        cache[A.shape[0]] = det(A[:-1, :-1]) * A[-1, -1] - A[0, 1] * A[1, 0] * det(A[:-2, :-2])
        return cache[A.shape[0]]
    return det(A)


print(np.linalg.det(A))
print(cheap_det(A))
