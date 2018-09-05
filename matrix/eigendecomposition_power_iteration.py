import numpy as np

# if not symmetric, then removing the maximal eigenvalue will modify other eigenvectors, breaking the eigendecomposition
symmetric = False
iterations = 10000
size = 5


A = np.random.uniform(size=(size, size))

if symmetric:
    A = A + A.T


# diagonalizes A
def eig(A, rank=None):
    # assume A is full rank
    if rank is None:
        rank = A.shape[0]

    # ignore zero eigenvalues
    if rank is 0:
        return [], []

    point = np.random.uniform(-1, 1, size=A.shape[1])
    point /= np.sqrt(point @ point)

    for i in range(iterations):
        point = A @ point
        point /= np.sqrt(point @ point)

    # Rayleigh quotient to derive eigenvalue from the eigenvector
    eigvec = point
    eigval = point @ A @ point / (point @ point)

    # set primary eigenvalue to zero
    test_eigvals, test_eigvecs = np.linalg.eig(A)
    test_eigvals[np.argmax(test_eigvals)] = 0

    B = test_eigvecs.T @ np.diag(test_eigvals) @ np.linalg.inv(test_eigvecs.T)

    # perform an eigendecomposition on the resultant rank n - 1 matrix
    eigvals, eigvecs = eig(B, rank - 1)

    return np.array([eigval, *eigvals]), np.array([eigvec, *eigvecs])


# sort eigvalues and eigvectors by the eigenvalues
def eig_sort(eigvals, eigvecs):
    return np.sort(eigvals), np.array([vec for val, vec in sorted(zip(eigvals, eigvecs))])


# using the recursive power iteration method
eigvals, eigvecs = eig_sort(*eig(A))
print(eigvals)

# compare against numpy's implementation
eigvals_np, eigvecs_np = np.linalg.eig(A)
eigvals_np, eigvecs_np = eig_sort(eigvals_np, eigvecs_np.T)

# all eigenvalues are same
print(np.allclose(eigvals, eigvals_np))


# all absolute eigenvectors are same
print(np.allclose(abs(eigvecs), abs(eigvecs_np)))
