import cv2
import requests
import os
import numpy as np

from sklearn.utils.extmath import randomized_svd

temp_directory = os.path.join(os.getcwd(), 'temp')
quantized_directory = os.path.join(os.getcwd(), 'quantizedImages')
compressed_directory = os.path.join(os.getcwd(), 'compressedImages')

cluster_counts = [5, 15, 30]
component_counts = [10, 50, 100]

image_urls = [
    # Add urls to images here:
]

image_paths = [os.path.join(temp_directory, os.path.basename(url)) for url in image_urls]

if not os.path.exists(temp_directory):
    os.makedirs(temp_directory)

for image_url in image_urls:
    image_path = os.path.join(temp_directory, os.path.basename(image_url))
    if not os.path.exists(image_path):
        print(f'Downloading: {image_url}')
        with open(image_path, 'wb') as image_file:
            for chunk in requests.get(image_url, stream=True):
                image_file.write(chunk)


# NOTE: saving images to an indexed color mode PNG is likely the most efficient data structure for quantized images
def image_quantize(image, num_colors):
    _, classified, means = cv2.kmeans(image.reshape((-1, 3)).astype(np.float32), K=num_colors, bestLabels=None,
                                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10), attempts=1,
                                      flags=cv2.KMEANS_RANDOM_CENTERS)

    return means.astype(np.uint8)[classified[:, 0]].reshape(image.shape)


# The PCA is with respect to the covariance matrix of each column with each column; equivalently SVD on the image
# randomized_svd: compute the k largest singular values and their accompanying k left and right singular vectors
#   to efficiently compute the top k eigenpairs of the eigendecomposition of the covariance matrix,

# images saved in the compressedImages folder are after recombining the decomposed image portions
# new sizes for the compressed images are determined from the number of bytes needed to store the scores and axes
#   (derived from the decomposition) into JPEG. This is the size printed in the console
def image_pca(image, K):
    processed = image.astype(np.float32)

    shift, scale = processed.mean(), processed.std() / np.sqrt(image.shape[0] * image.shape[1])
    # shift, scale = 1, 1
    processed = (processed - shift) / scale

    total_bytes = 0
    channels = []
    for channel in range(3):
        U, S, Vt = randomized_svd(processed[..., channel], n_components=K)
        scores = U @ np.diag(S)
        compressed = (scores / (np.max(scores) - np.min(scores)) * 128 + 128).astype(np.uint8)
        total_bytes += cv2.imencode('.jpg', compressed)[1].nbytes
        scores = (compressed.astype(np.float32) - 128) / 128 * (np.max(scores) - np.min(scores))

        axes = Vt
        compressed = (axes / (np.max(axes) - np.min(axes)) * 128 + 128).astype(np.uint8)
        total_bytes += cv2.imencode('.jpg', compressed)[1].nbytes
        axes = (compressed.astype(np.float32) - 128) / 128 * (np.max(axes) - np.min(axes))

        channels.append(scores @ axes)
    print(f'{int(total_bytes / 1024 + 1)}Kb')

    processed = np.dstack(channels) * scale + shift
    return np.clip(processed, 0, 255).astype(np.uint8)


def demonstration(func, params, path):
    for image_path in image_paths:
        image = cv2.imread(image_path)

        for param in params:
            print(f'{os.path.basename(image_path)} {func.__name__} {param}')
            reduced = func(image, param)
            cv2.imshow('preview', reduced)

            cv2.waitKey(1000)

            if param == params[1]:
                if not os.path.exists(path):
                    os.mkdir(path)

                cv2.imwrite(os.path.join(path, str(param) + '_' + os.path.basename(image_path)), reduced)


demonstration(image_quantize, cluster_counts, quantized_directory)
demonstration(image_pca, component_counts, compressed_directory)
