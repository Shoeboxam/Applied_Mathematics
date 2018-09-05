import numpy as np

# functions to create elementary matrices for gaussian elimination row ops

size = 5
u = np.random.uniform(-1, 1, size=(size, 1)) + np.random.uniform(-1, 1, size=(size, 1)) * 1j
v = np.random.uniform(-1, 1, size=(size, 1)) + np.random.uniform(-1, 1, size=(size, 1)) * 1j

A = np.eye(size) - u @ v.T
# all elementary matrices are invertible
print(np.allclose(np.linalg.inv(A), np.eye(size) - u @ v.T / (u.T @ v - 1)))


# elementary matrix row operations for RREF
def elementary_1(size):
    i, j = np.random.choice(range(size), size=2, replace=False)
    trans = np.eye(size)[i] - np.eye(size)[j]
    return np.eye(size) - np.einsum('i,j->ij', trans, trans)


def elementary_2(size, alpha):
    i = np.random.randint(size)
    return np.eye(size) - (1 - alpha) * np.einsum('i,j->ij', np.eye(size)[i], np.eye(size)[i])


def elementary_3(size, alpha):
    i, j = np.random.choice(range(size), size=2, replace=False)
    return np.eye(size) - alpha * np.einsum('i,j->ij', np.eye(size)[i], np.eye(size)[j])


print(elementary_3(size, .5))


# Set all rows but i equal to zero
def filter_row(A, i):
    return np.einsum('i,j,jk->ik', np.eye(A.shape[0])[i], np.eye(A.shape[0])[i], A)


print(filter_row(A, 2))
