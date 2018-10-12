import timeit
import numpy as np

# this is an algorithmic complexity comparison of vectorized matrix derivatives, vs tensor based derivatives

dimensions = [10, 100]

A = np.random.uniform(0, 1, size=dimensions)
b = np.random.uniform(0, 1, size=[dimensions[1], 1])
e = np.random.uniform(0, 1, size=[dimensions[0], 1])

# Function to be optimized, with respect to A
# Err = (e - Ab)^2


# matrix derivative with the kron product
def gradient_matrix():
    dErr_dAvec = (e - A @ b).T @ np.kron(b.T, np.eye(A.shape[0]))
    return np.reshape(dErr_dAvec, newshape=A.shape, order='F')  # vec inverse, requires FORTRAN order indexing


# tensor derivative is simpler, faster, and takes less memory O(n^2) vs O(n^3)
def gradient_tensor():
    return (e - A @ b) @ b.T


def time_test(trials):
    print("Matrix Derivative Time")  # 0.1345 seconds
    print(timeit.timeit('gradient_matrix()', 'from __main__ import gradient_matrix', number=trials))
    print("Tensor Derivative Time")  # 0.0034 seconds
    print(timeit.timeit('gradient_tensor()', 'from __main__ import gradient_tensor', number=trials))

    print("\nAre matrices the same?")
    print(np.allclose(gradient_matrix(), gradient_tensor()))


def vectorization_test():
    W = np.random.normal(size=(5, 4))
    s = np.random.uniform(size=(4, 2))

    method1 = np.kron(s.T, np.eye(W.shape[0])) @ np.reshape(W, (-1, 1), order='F')
    method2 = np.reshape(W @ s, (-1, 1), order='F')

    print('[s.T ⊗ I] vec(W) == Ws')
    print(np.allclose(method1, method2))


time_test(trials=1000)