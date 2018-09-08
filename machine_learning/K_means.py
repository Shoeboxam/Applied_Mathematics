from scipy import cluster
import numpy as np
import matplotlib.pyplot as plt

# Used for 3D plotting, but not directly invoked
from mpl_toolkits.mplot3d import Axes3D


# ~~~~ Generate a dataset ~~~~
# Generate a positive semidefinite covariance matrix for each cluster
rand = np.random.uniform(-1, 1, size=(3, 3))
covariance_1 = rand @ rand.T
source_one = np.random.multivariate_normal((2, 1, 8), covariance_1, size=40)

rand = np.random.uniform(-1, 1, size=(3, 3))
covariance_2 = rand @ rand.T
source_two = np.random.multivariate_normal((4, 0, 6), covariance_2, size=20)

# Combine the two point sets and shuffle their points (rows)
dataset = np.vstack((source_one, source_two))
np.random.shuffle(dataset)


# ~~~~ K-Means Clustering ~~~~
codebook, _ = cluster.vq.kmeans(dataset, 2, iter=20)
print("Centroids:")
print(codebook)

# Use the codebook to assign each observation to a cluster via vector quantization
labels, __ = cluster.vq.vq(dataset, codebook)

# Use boolean indexing to extract points in a cluster from the dataset
cluster_one = dataset[labels == 0]
cluster_two = dataset[labels == 1]

# Check number of nodes assigned to a cluster
# print(np.shape(cluster_one)[0])


# ~~~~ Visualization ~~~~
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(*codebook.T, c='r')
ax.scatter(*cluster_one.T, c='b', s=3)
ax.scatter(*cluster_two.T, c='g', s=3)

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

plt.show()
