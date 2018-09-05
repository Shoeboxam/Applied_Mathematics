import numpy as np
from scipy import stats


# partition A into bins by a list of edges
def partition(A, edges):
    if len(edges):
        return [A[A <= edges[0]], *partition(A[A > edges[0]], edges[1:])]
    return [A]


# under the assumption of distribution, with N samples, how many samples are expected to be in each bin
def expectation(N, distribution, edges):
    expectations = []
    for edge in edges:
        expectations.append(distribution.cdf(edge) * N - np.sum(expectations))
    expectations.append(N - np.sum(expectations))
    return expectations


def chisq_stat(actual, expected):
    return np.sum(np.array((actual - expected) ** 2 / expected))


sample = np.array([
    -2.4347, -2.3361, -2.1925, -2.0100, -1.9670, -1.7076, -1.6780, -1.5634, -1.4763, -1.3886, -1.3318, -1.2695,
    -1.2295, -1.2270, -1.1746, -1.1361, -1.1277, -1.1249, -1.1204, -1.0732, -1.0526, -1.0513, -1.0321, -0.9389,
    -0.8842, -0.8479, -0.8464, -0.7160, -0.6448, -0.6253, -0.5880, -0.5847, -0.4961, -0.4892, -0.4731, -0.4536,
    -0.4279, -0.3958, -0.3864, -0.3860, -0.3737, -0.3440, -0.2800, -0.2467, -0.2397, -0.2117, -0.1883, -0.1557,
    -0.1495, -0.1129, -0.1031, -0.1010, -0.0331, -0.0115, 0.0338, 0.1107, 0.1396, 0.1434, 0.2185, 0.2186, 0.2519,
    0.2618, 0.3085, 0.3438, 0.3578, 0.4633, 0.4773, 0.4820, 0.4898, 0.5453, 0.5906, 0.6382, 0.6520, 0.6560, 0.6730,
    0.7726, 0.7758, 0.7765, 0.7879, 0.9697, 0.9789, 1.0051, 1.0130, 1.0393, 1.0726, 1.1680, 1.1856, 1.2630, 1.2696,
    1.2975, 1.3603, 1.3708, 1.6817, 1.7210, 1.7355, 1.7792, 1.7925, 1.8812, 1.9036, 2.0098
])


# ~~~~ Chisq Normality Test ~~~~
norm_edges = [-1.5, -.5, 0, .5, 1.5]

actual = np.array([len(part) for part in partition(sample, norm_edges)])
expected = expectation(len(sample), stats.norm(0, 1), norm_edges)

print("P-values for chisq test for normality")
# conduct a chisq test to get p-value manually
test_statistic = chisq_stat(actual, expected)
print(1 - stats.chi2(5).cdf(test_statistic))

# alternatively, use built in chisq test
print(stats.chisquare(actual, expected).pvalue)


# ~~~~ Uniform Test ~~~~

uniform_edges = [-2, -1, 0, 1, 2]

actual = np.array([len(bin) for bin in partition(sample, uniform_edges)])
expected = expectation(len(sample), stats.uniform(-3, 6), uniform_edges)

print("P-values for chisq test for uniformity")

test_statistic = chisq_stat(actual, expected)
print(1 - stats.chi2(5).cdf(test_statistic))

print(stats.chisquare(actual, expected).pvalue)
