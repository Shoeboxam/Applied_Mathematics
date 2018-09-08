from math import cos, sin, pi
import numpy as np

# generates a cayley table for the group of dihedral permutations with arbitrary number of edges
# https://en.wikipedia.org/wiki/Dihedral_group

# generate cayley table for n-gon with deg edges
deg = 4

rotations = []
mirrors = []

for i in range(deg):
    rotat = np.array([[cos(2 * pi * i / deg), -sin(2 * pi * i / deg)],
                      [sin(2 * pi * i / deg), cos(2 * pi * i / deg)]])
    mirro = np.array([[cos(2 * pi * i / deg), sin(2 * pi * i / deg)],
                      [sin(2 * pi * i / deg), -cos(2 * pi * i / deg)]])

    rotations.append(rotat)
    mirrors.append(mirro)

transformations = rotations + mirrors

cayley = []
for i in range(len(transformations)):
    row = []
    for j in range(len(transformations)):
        product = np.matmul(transformations[i], transformations[j])
        for k in range(len(transformations)):
            if np.allclose(transformations[k], product, rtol=.00001):
                row.append(k)
                continue

    cayley.append(row)

print(np.array(cayley))
