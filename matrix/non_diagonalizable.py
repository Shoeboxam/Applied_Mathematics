import numpy as np

# since numpy uses finite-precision floats, non-diagonal matrices cannot be represented
# All matrices, even ones constructed to be non-diagonal, will be diagonalizable via eigendecomposition

A = np.array([[1, 1], [1, 0]])
eigvals, eigvecs = np.linalg.eig(A)

print(A)
print(eigvecs @ np.diag(eigvals) @ np.linalg.inv(eigvecs))


