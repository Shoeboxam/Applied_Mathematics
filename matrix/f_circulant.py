import numpy as np


# the characteristic equations of f circulants form a polynomial ring
def f_circulant(f, size):
    return np.block([[np.zeros(shape=[1, size - 1]), f],
                     [np.eye(size - 1), np.zeros(shape=[size - 1, 1])]])


F = f_circulant(2, 5) @ f_circulant(3, 5) @ f_circulant(4, 5)

print(F)
print(np.linalg.matrix_power(F, 5))
