import numpy as np


# make a matrix of size s with rank r
def make_rank(s, r):
    return np.random.uniform(size=(s, r)) @ np.random.uniform(size=(r, s))


A = make_rank(5, 1)
B = make_rank(5, 3)

# There are several rank properties:
#   rank(AB) <= min(rank(A), rank(B))
#   rank(A) + rank(B) - n <= rank(AB) Sylvester

print(np.linalg.matrix_rank(A))
print(np.linalg.matrix_rank(B @ A))
