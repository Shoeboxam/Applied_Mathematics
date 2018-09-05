import numpy as np

# example of typical matrix multiplication conducted over block submatrices

dims = np.random.randint(5, 10, size=3)

A = np.random.uniform(size=dims[:2])
B = np.random.uniform(size=dims[1:])

x, y, z = np.random.randint(1, 5, size=3)

A_split = [
    [A[:y, :x], A[:y, x:]],
    [A[y:, :x], A[y:, x:]]
]

B_split = [
    [B[:x, :z], B[:x, z:]],
    [B[x:, :z], B[x:, z:]]
]

C_blocked = np.block([
    [A_split[0][0] @ B_split[0][0] + A_split[0][1] @ B_split[1][0],
     A_split[0][0] @ B_split[0][1] + A_split[0][1] @ B_split[1][1]],

    [A_split[1][0] @ B_split[0][0] + A_split[1][1] @ B_split[1][0],
     A_split[1][0] @ B_split[0][1] + A_split[1][1] @ B_split[1][1]]
])

C = A @ B

print(np.allclose(C, C_blocked))
