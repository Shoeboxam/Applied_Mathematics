import numpy as np

# WARNING: the Laplace expansion has O(n!) time complexity
size = 6


# https://en.wikipedia.org/wiki/Laplace_expansion
def det(A):
    if A.shape == (1, 1):
        return A[0, 0]
    return sum([(i % 2 * -2 + 1) * A[0, i] * minor(A, [0, i]) for i in range(A.shape[0])])


# determinant of A with row and column removed
def minor(A, indices):
    for axis, index in enumerate(indices):
        A = np.delete(A, index, axis)
    return det(A)


# https://en.wikipedia.org/wiki/Adjugate_matrix
def adjugate(A):
    sign_mask = np.add.outer(*[np.array(range(i)) for i in A.shape]) % 2 * -2 + 1
    return sign_mask * np.array([[minor(A, [i, j]) for j in range(A.shape[1])] for i in range(A.shape[0])])


# compute inverse of A at indices i, j
def inv_partial(A, indices):
    return -1 ** sum(indices) * minor(A, reversed(indices)) / det(A)


A = np.random.uniform(size=(size, size))

print("Equal determinant:")
print(np.allclose(np.linalg.det(A), det(A)))

# compute matrix inverse via transpose of adjugate matrix, divided by determinant
print("Equal matrix inverse:")
print(np.allclose(np.linalg.inv(A), adjugate(A).T / np.linalg.det(A)))

print("Matrix inverse at a single point:")
print(np.allclose(np.linalg.inv(A)[2, 3], inv_partial(A, [2, 3])))
