import numpy as np

# can solve linear systems in xA or Ax form via transposition
size = 5
y = np.random.uniform(size=[size] * 2)
x = np.random.uniform(size=[size] * 2)

A_right = np.linalg.solve(x, y)
print(np.allclose(x @ A_right, y))  # xA = y

A_left = np.linalg.solve(x.T, y.T).T
print(np.allclose(A_left @ x, y))  # Ax = y


# inverses can be computed for non-square or rank-deficient matrices
A = np.random.uniform(size=(5, 3))
b = np.random.uniform(size=(5, 1))
print(np.linalg.lstsq(A, b, rcond=None))
