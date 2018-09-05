import numpy as np
from scipy import linalg

# demonstrate differences in numpy/scipy eigdecomposition,
# and the biorthogonal system constructed via left/right decompositions

np.set_printoptions(suppress=True, precision=3)

size = 5
A = np.random.uniform(size=(size, size))


# sort eigvalues and eigvectors by the eigenvalues
def eig_sort(eigvals, eigvecs):
    return np.sort(eigvals), np.array([vec for val, vec in sorted(zip(eigvals, eigvecs))])


# numpy returns right eigenvectors vA = λv. Eigenvectors are rows
numpy_eigval, numpy_right_vec = eig_sort(*np.linalg.eig(A))

scipy_eigval, scipy_left_vec, scipy_right_vec = linalg.eig(A, left=True, right=True)
scipy_left_vec = eig_sort(scipy_eigval, scipy_left_vec)[1]  # Av = λv
scipy_right_vec = eig_sort(scipy_eigval, scipy_right_vec)[1]  # vA = λv

print(np.allclose(numpy_right_vec, scipy_right_vec))


# The left and right eigenvectors are bases for dual spaces, and together form a biorthogonal system equivalent to δ (I)
# In practice this doesn't actually give δ, possibly due to numerical instability?
# https://en.wikipedia.org/wiki/Biorthogonal_system
print(scipy_left_vec.conj().T @ scipy_right_vec)
