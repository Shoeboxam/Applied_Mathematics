import numpy as np
import pandas as pd

np.set_printoptions(suppress=True, precision=4)

samples = 1000


def get_data(id):
    if id is 'normal':
        return np.random.multivariate_normal([0, 0, 0], np.diag([2, 3, 5]), size=samples)

    if id is 'uniform':
        bounds = np.array([30, 15, 24])
        return np.random.uniform(bounds, -bounds, size=[samples, bounds.size])

    if id is 'dataset':
        raw_data = pd.read_csv('../datasets/regression/APS Failure at Scania Trucks/aps_failure_training_set.csv')

        columns = ['aa_000', 'ac_000', 'aq_000']

        for column in columns:
            raw_data[column] = pd.to_numeric(raw_data[column], errors='coerce')

        return np.array(raw_data[columns].dropna()[:samples])


data = get_data('normal')

print('correlation matrix')
print(np.corrcoef(data.T))

# convert covariance matrix to correlation matrix:
covariances = np.cov(data.T)

print('correlation matrix derived from covariance matrix')
print(np.diag(np.diag(covariances) ** (-1/2)) @ covariances @ np.diag(np.diag(covariances) ** (-1/2)))

# standardize dataset, then covariance matrix is the correlation matrix for the original dataset

variances = np.var(data, axis=0)
data_standard = data / np.sqrt(np.var(data, axis=0, ddof=1))

print('covariance matrix of standardized data')
print(np.cov(data_standard.T))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('\n\nMLE estimator for covariance matrix')

data_centered = data - np.mean(data, axis=0)
print('covariance matrix from numpy')
print(np.cov(data_centered.T))

print('covariance matrix derived, where X.T @ X ~ W(∑, n-1)')
print(data_centered.T @ data_centered / (data_centered.shape[0] - 1))
