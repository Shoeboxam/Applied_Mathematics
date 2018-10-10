import numpy as np

from sklearn.decomposition import PCA

np.set_printoptions(suppress=True, precision=4)

attributes = 3
samples = 1000


# make eigendecompositions and singular value decompositions deterministic by:
# 1. remove singular/eigen vectors with zero singular/eigen values
# 2. make largest value in each left vector positive by flipping sign of corresponding right vector
# 3. sort singular/eigen vectors by magnitudes of singular/eigen values
def deterministic(U, S, V=None):
    # find the sign corrections needed for the left vectors
    signs = np.sign(U[np.argmax(np.abs(U[:, :S.size]), axis=0), np.arange(S.size)])

    # find indexes of elements in descending order of magnitude
    order = np.argsort(-S)

    # truncate, correct sign, then sort
    return (U[:, :S.size] * signs)[:, order], S[order], V is None or (V[:S.size] * signs[..., None])[order]


def pca_example(data):

    U, S, V = deterministic(*np.linalg.svd(data - np.mean(data, axis=0)))

    print('\n\nEquivalent SVD after adjusting order/vectors/signs?')
    print(np.allclose(data, U @ np.diag(S) @ V))

    pca = PCA(n_components=3)
    pca.fit(data)

    print('PCA components_ and the deterministic right singular vectors from numpy are equal?')
    print(np.allclose(pca.components_, V))

    print('Singular values are equal?')
    print(np.allclose(pca.singular_values_, S))


base = np.random.uniform(-5, 5, size=[attributes] * 2)
covariance = base.T @ base
mean = np.random.uniform(-10, 10, size=attributes)

print('mean')
print(mean)

print('covariance matrix')
print(covariance)

data = np.random.multivariate_normal(mean, covariance, size=samples)
pca_example(data)
