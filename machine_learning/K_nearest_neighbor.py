import numpy as np

k = 5

n_1 = 40
n_2 = 20

mean_0 = np.array([2, 1, 8])
mean_1 = np.array([4, 0, 6])

dims = mean_0.size

# ~~~~ Generate a dataset ~~~~
# Generate a positive semidefinite covariance matrix for each cluster
rand = np.random.uniform(-1, 1, size=[dims] * 2)
covariance_1 = rand @ rand.T
source_one = np.random.multivariate_normal(mean_0, covariance_1, size=n_1)
source_one = np.block([source_one, np.full(shape=(n_1, 1), fill_value=0)])

rand = np.random.uniform(-1, 1, size=[dims] * 2)
covariance_2 = rand @ rand.T
source_two = np.random.multivariate_normal(mean_1, covariance_2, size=n_2)
source_two = np.block([source_two, np.full(shape=(n_2, 1), fill_value=1)])

# Combine the two point sets and shuffle their points (rows)
dataset = np.vstack((source_one, source_two))
np.random.shuffle(dataset)


def p_norm(x, point, p=2):
    return np.sum((x - point) ** p, axis=1) ** (1 / p)


distances = p_norm(dataset[:, :dims], mean_1)
dataset_by_distance = dataset[np.argsort(distances)]
print(np.mean(dataset_by_distance[:k, -1]))
