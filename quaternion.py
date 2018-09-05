import numpy as np
# pip install pyquaternion
from pyquaternion import Quaternion

np.set_printoptions(suppress=True, precision=3)

# to be considered rotations, they must be points on the unit 4-sphere, so normalize
q1 = Quaternion(1, 2, -1, 4).normalised
q2 = Quaternion(2, 5, -2, 1).normalised

print("quaternion multiplication produces the same rotation as successive rotation matrix products")
print(np.allclose((q1 * q2).rotation_matrix, q1.rotation_matrix @ q2.rotation_matrix))

# test quaternion
qi = Quaternion(1, 0, 1, 0).normalised
A = qi.transformation_matrix

# construct a quaternion basis set
# https://core.ac.uk/download/pdf/82668219.pdf
B = np.array([
    [0, 1],
    [-1, 0]
])

C = np.array([
    [0, 1],
    [1, 0]
])

D = np.array([
    [1, 0],
    [0, -1]
])

E = np.array([
    [1, 0],
    [0, 1]
])

# matrix bases isomorphic to quaternion units
I = np.kron(E, E)
H = np.kron(D, B)
J = np.kron(B, E)
K = np.kron(C, B)
quaternion_base = np.array([I, H, J, K])

# TODO: doesn't work yet
print(sum(quaternion_base * qi.elements[:, None, None]))
print(qi.transformation_matrix)