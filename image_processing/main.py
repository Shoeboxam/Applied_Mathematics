import numpy as np
from PIL import Image

# openCV package is distributed as a binary, so references won't resolve
import cv2

from image_processing.canny import canny_mask
from image_processing.difference_of_gaussians import difference_of_gaussians

use_landscape = True


def test_canny(im):
    im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    mask, (y, x, h, w) = canny_mask(im, debug=True)

    # Apply mask, then slice to bounding box of mask
    mask = np.dstack([mask] * 3).astype('float32') / 255.0
    im = (mask * im.astype('float32')).astype('uint8')  # [x:x + w, y:y + h] # cropping disabled

    # Convert back to PIL
    im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 'RGB')

    cv2.imshow("processed", cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
    cv2.waitKey()


def test_difference_of_gaussian(im, depth, bound):
    image = difference_of_gaussians(im, depth, bound - depth)

    cv2.imshow("processed", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    if depth == bound: return


if __name__ == '__main__':
    # run on the image in the path:
    path = 'Bikesgray.jpg'

    # Load image into array
    data = np.array(Image.open(path)).astype(float)

    depth = 0
    while True:
        # progressive image segmentation
        bound = 100
        test_difference_of_gaussian(data, depth, bound)
        depth += 10
        if depth == bound:
            break
