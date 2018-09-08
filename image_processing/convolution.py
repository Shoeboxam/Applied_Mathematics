from PIL import Image
import numpy as np

sharpen = np.array(
    [[+0, -1, 0],
     [-1, +5, -1],
     [+0, -1, 0]])

laplacian = np.array(
    [[+0, +1, +0],
     [+1, -4, +1],
     [+0, +1, +0]])

gaussian = np.array(
    [[1, 2, 1],
     [2, 4, 2],
     [1, 2, 1]]) / 16

identity = np.array(
    [[0, 0, 0],
     [0, 1, 0],
     [0, 0, 0]])

edge = np.array(
    [[+0, 0, 0],
     [-1, 0, 1],
     [+0, 0, 0]])

emboss = np.array(
    [[-2, -1, 0],
     [-1, 1, 1],
     [+0, 1, 2]])


def convolve(A, kernel):
    # Determine edge of kernel
    kernel_edge = np.array(kernel.shape) - 1
    if len(A.shape) == 3:
        kernel_edge = np.append(kernel_edge, np.array([0]))

    # Convert 3D image into 3D sampler [kernel elements, channel, sample index]
    shape = [*kernel.shape, *(A.shape - kernel_edge)]
    memory_offset = [*A.strides[:2], *A.strides]
    samples = np.lib.stride_tricks.as_strided(A, shape=shape, strides=memory_offset)

    # Contract over first two dimensions (note that samples is a 5-dimensional array)
    return np.einsum("ij...,ij->...", samples, kernel)


def sobel(A):
    sobel_x = np.array(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]])

    sobel_y = np.array(
        [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]])

    return np.sqrt(convolve(A, sobel_x)**2 + convolve(A, sobel_y)**2)


if __name__ == '__main__':
    # run a filter over the image in the path:
    path = 'Bikesgray.jpg'

    # Load image into array
    data = np.array(Image.open(path)).astype(float)

    # Apply operator
    convolved = convolve(data, emboss)
    # convolved = sobel(data)

    # Convert back to image and show
    Image.fromarray(np.clip(convolved, 0, 255).astype(np.uint8)).show()
