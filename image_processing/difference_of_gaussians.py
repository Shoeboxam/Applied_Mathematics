import numpy as np
from PIL import Image
from PIL import ImageFilter

from image_processing.convolution import convolve, laplacian


def gabor_fn(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb


def difference_of_gaussians(image, lowpass, highpass):
    # Load image into array
    # newsize = (np.array(image.size) / image.size[0] * 400).astype(int)
    # data = np.array(image.resize(newsize, Image.ANTIALIAS)).astype(float)

    max_blurred = image.copy()
    min_blurred = image.copy()
    for depth in range(max(lowpass, highpass)):
        max_blurred = max_blurred.filter(ImageFilter.BLUR)
        if depth == min(lowpass, highpass):
            min_blurred = max_blurred.copy()

    if lowpass < highpass:
        image_lowpass, image_highpass = min_blurred, max_blurred
    else:
        image_lowpass, image_highpass = max_blurred, min_blurred

    processed = np.array(image_lowpass) - np.array(image_highpass)

    processed = convolve(processed.astype(float), laplacian)

    # print(np.array(image_lowpass).shape)
    # image_lowpass.show()
    # image_highpass.show()

    # print(edges)
    return Image.fromarray(np.clip(processed + 128, 0, 255).astype(np.uint8), 'RGB')

    # gabor_fn(np.array(Image.open('etc/sample_1.jpg'))
