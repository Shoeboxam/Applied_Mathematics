import numpy as np


def make_clusters(parameters, labels=False):

    points = []
    for i, params in enumerate(parameters):

        # Generate a positive semidefinite covariance matrix
        if 'covariance' in params:
            covariance = params['covariance']
        else:
            rand = np.random.uniform(-1, 1, size=[len(params['mean'])] * 2)
            covariance = rand @ rand.T

        num_points = params.get('n', 20)

        cluster_points = np.random.multivariate_normal(params['mean'], covariance, size=num_points)

        if labels:
            cluster_points = np.block([cluster_points, np.full(shape=(num_points, 1), fill_value=i)])

        points.append(cluster_points)

        # Combine the two point sets and shuffle their points (rows)
    dataset = np.vstack(points)
    np.random.shuffle(dataset)

    return dataset
