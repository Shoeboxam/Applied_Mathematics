import numpy as np
from scipy import stats

import pandas as pd


def chisq_stat(actual, expected):
    return np.sum(np.array((actual - expected) ** 2 / expected))


actual = pd.DataFrame(np.array([[10, 30], [10, 50]]), columns=['Yes', 'No'], index=['Black', 'White'])
expected = np.outer(np.sum(actual, axis=1), np.sum(actual, axis=0)) / np.sum(np.array(actual))

test_statistic = chisq_stat(actual, expected)

print("manual chi2 test for independence")
print(1 - stats.chi2(1).cdf(test_statistic))

print("using built-in one-way chi2 test")
print(stats.chisquare(np.reshape(np.array(actual), [-1]), np.reshape(expected, [-1]), ddof=2).pvalue)

print("using built-in contingency table chi2 test")
# since DOF is 1, Yates correction is applied by default
print(stats.chi2_contingency(actual, correction=False))
