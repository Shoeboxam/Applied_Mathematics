from scipy import cluster
from machine_learning.utilities import make_clusters
import matplotlib.pyplot as plt

# Used for 3D plotting, but not directly invoked
from mpl_toolkits.mplot3d import Axes3D

# ~~~~ Generate a dataset ~~~~
dataset = make_clusters([{'mean': [2, 1, 8], 'n': 20}, {'mean': [4, 0, 6], 'n': 40}])


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
