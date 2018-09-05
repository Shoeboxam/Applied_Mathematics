import numpy as np
from scipy import linalg
import math

A = np.random.uniform(size=(5, 5))


# implementations by Taylor series expansions
def exp_mat(A, n):
    return sum([np.linalg.matrix_power(A, i) / math.factorial(i) for i in range(n)])


# TODO: this is still not correct
def log_mat(A, n):
    return sum([-1 ** (i + 1) * np.linalg.matrix_power(A - np.eye(A.shape[0]), i) / i for i in range(1, n + 1)])


print(np.allclose(exp_mat(A, 20), linalg.expm(A)))
print(np.allclose(log_mat(A, 20), linalg.logm(A)))

print(log_mat(A, 20))
print(linalg.logm(A))