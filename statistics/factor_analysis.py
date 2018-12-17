# implement loadings for principal component factor analysis
# compare against loadings from principal factor, factor analysis from sklearn
# compare varimax transformed loadings


# Equivalence to R:
# the principal component factor analysis returns equivalent values to R Psych principal
# the principal factor, factor analysis returns equivalent values to R Psych factanal

# library(psych)
# print(factanal(data_subset, factors=2, rotation="varimax"))
# print(principal(data_subset, nfactors=1, rotate="none"))


import numpy as np
import pandas as pd
import requests
import zipfile
import io

from sklearn.decomposition import FactorAnalysis
from sklearn import preprocessing

np.set_printoptions(suppress=True, precision=4)

scale = True

# data = np.array(pd.read_csv('./air_data.csv'))


remote_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip'
remote = io.BytesIO(requests.get(remote_url, stream=True).content)

with zipfile.ZipFile(remote, 'r') as zipped:
    data = pd.read_csv(zipped.open('AirQualityUCI.csv'), delimiter=';')

data = data[['PT08.S1(CO)', 'NMHC(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)']]
data = data.dropna().values


# make eigendecompositions and singular value decompositions deterministic by:
# 1. remove singular/eigen vectors with zero singular/eigen values
# 2. make largest value in each left vector positive by flipping sign of corresponding right vector
# 3. sort singular/eigen vectors by magnitudes of singular/eigen values
def deterministic(U, S, Vt=None):
    # find the sign corrections needed for the left vectors
    signs = np.sign(U[np.argmax(np.abs(U[:, :S.size]), axis=0), np.arange(S.size)])

    # find indexes of elements in descending order of magnitude
    order = np.argsort(-S)

    # truncate, correct sign, then sort
    return (U[:, :S.size] * signs)[:, order], S[order], Vt is None or (Vt[:S.size] * signs[..., None])[order]


def analyze_components(data):
    data = data - data.mean(axis=0)
    if scale:
        data /= data.std(axis=0)

    U, S, Vt = deterministic(*np.linalg.svd(data))
    print('Principal Components:')
    print(Vt)

    variances = S**2

    # equivalent method to compute variances via eigendecomposition
    # variances = np.sort(np.linalg.eig(np.cov(data.T))[0])[::-1]

    # check that sum of sample variances is equal to sum of eigvals
    # print(np.var(data, axis=0, ddof=1).sum() - variances.sum())

    ratios = variances / variances.sum()
    print('Ratio:     ', ratios)
    print('Cumulative:', ratios.cumsum())

    try:
        import matplotlib.pyplot as plt
        plt.plot(range(1, len(variances) + 1), variances)
        # plt.show()
    except ImportError:
        print('pip install matplotlib for a scree plot')

    print('\nKaiser criterion suggests', len(variances[variances > variances.mean()]), 'components\n')


def loadings(data, m, communalities=None):
    if communalities is None:
        communalities = np.ones(data.shape[1])

    # form the reduced correlation matrix
    corr = np.corrcoef(data.T)
    corr[np.diag_indices(corr.shape[0])] = communalities

    # get top m eigvals/eigvecs
    eigvals, eigvecs = np.linalg.eig(corr)
    sort_indexes = eigvals.argsort()[::-1][:m]
    eigvals, eigvecs = eigvals[sort_indexes], eigvecs[:, sort_indexes]

    # final loadings
    return eigvecs @ np.diag(eigvals**(1/2))


def principal_factor_solution(L):
    return np.ones(L.shape[0]) - np.diag(L @ L.T)


# adapted from: https://en.wikipedia.org/wiki/Talk:Varimax_rotation, same variable names used from Lecture 9 notes
def varimax(L, gamma=1, lim=20, tol=1e-6):
    p, m = L.shape

    # rotation matrix that is iteratively updated
    T = np.eye(m)

    d = 0
    for i in range(lim):
        LT = L @ T
        temp = LT ** 3 - (gamma / p) * LT @ np.diag(np.diag(LT.T @ LT))
        U, S, Vh = np.linalg.svd(L.T @ temp)

        T = U @ Vh
        d_old, d = d, np.sum(S)

        if d / d_old < tol:
            break

    return L @ T


# compute the sum of the variances of the loadings
def variance_metric(L):
    communalities = np.sum(L**2, axis=1)
    L_scaled = L / np.sqrt(communalities)[..., None]
    return np.sum(np.sum(L_scaled**4, axis=0) - np.sum(L_scaled**2, axis=0)**2 / L.shape[0]) / L.shape[0]


analyze_components(data)


print('Sample covariance matrix:')
data_centered = data - data.mean(axis=0)
print(data_centered.T @ data_centered / (data.shape[0] - 1))

print('Principal Component FA, M=1')
L_1 = loadings(data, m=1)

print('Solution:', principal_factor_solution(L_1))
print('Loading variances:', variance_metric(L_1))
print(L_1)

L_1 = varimax(L_1)
print('Loading variances (varimax):', variance_metric(L_1))
print(L_1)


print('Principal Component FA, M=2')
L_2 = loadings(data, m=2)

print('Solution:', principal_factor_solution(L_2))
print('Loading variances:', variance_metric(L_2))
print(L_2)

L_2 = varimax(L_2)
print('Loading variances (varimax):', variance_metric(L_2))
print(L_2)


fa = FactorAnalysis(n_components=2, tol=.000001, max_iter=10000)
fa.fit(preprocessing.scale(data))

L_2_F = fa.components_.T
print('Principal Factor Analysis, M=2')
print('Loading variances:', variance_metric(L_2_F))
print(L_2_F)

L_2_F = varimax(L_2_F)
print('Loading variances (varimax):', variance_metric(L_2_F))
print(L_2_F)
