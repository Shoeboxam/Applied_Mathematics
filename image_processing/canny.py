import numpy as np

# openCV package is distributed as a binary, so references won't resolve
import cv2


def canny_mask(img, debug=False):

    if debug:
        cv2.imshow("original", img)
    # Operate on greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection, thresholded between 10 and 200
    edges = cv2.Canny(gray, 10, 200)

    # Dilation then erosion to clean noising in edges, called closing. No kernel is used
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, None)

    # At this stage, the image is b/w, with white along edges, black within regions

    # ~~~ Now find regions, which are enclosed by contours:
    # Retrieval mode - consider landlocked contours children. TREE preserves hierarchy, LIST flattens
    # Chain approx - how many points used along edge of region.
    #                NONE is all, SIMPLE works well for straight edges, Teh-Chin for lossy fitting of a curvy contour
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # contours is a list of regions, each region is a list of boundary coordinates
    # hierarchy is a list of region metadata: [next sibling ID, previous sibling ID, child ID, parent ID]

    # Additional edge plot for seeing top n classifications
    if debug:
        contours = sorted(contours, key=lambda cont: cv2.contourArea(cont), reverse=True)[:5]

        tempmask = np.zeros(edges.shape)
        for idx, contour in enumerate(contours):
            cv2.fillPoly(tempmask, [contour], 1 - 1. / len(contours) * (idx + 1))
        cv2.imshow('mask', tempmask)

    # Catch a few cases where segmentation breaks down
    if not contours:
        if debug:
            raise ValueError("No contours detected")
        else:
            return np.ones(img.shape[:2]) * 255, (0, 0, *img.shape[:2])

    # Pick the contour with the greatest area, tends to represent the clothing item
    max_contour = max(contours, key=lambda cont: cv2.contourArea(cont))
    if not (.20 < cv2.contourArea(max_contour) / np.prod(img.shape[:2]) < .80):
        if debug:
            raise ValueError("Detected poor area coverage")
        else:
            return np.ones(img.shape[:2]) * 255, (0, 0, *img.shape[:2])

    # Create empty mask, draw filled polygon on it corresponding to largest contour
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillPoly(mask, [max_contour], 255)

    # Smooth mask, then blur it
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    # Catch another case where segmentation breaks down
    border_size = np.sum(img.shape[:2] * 2) - 2
    border_coverage = border_size - (np.sum(mask[-1:] + mask[:1]) + np.sum(mask[:, -1:] + mask[:, :1])) / 255
    if (border_coverage / border_size) < .6:
        if debug:
            raise ValueError("Detected poor border coverage")
        else:
            return np.ones(img.shape[:2]) * 255, (0, 0, *img.shape[:2])

    # First remove some fine details from the mask
    blur_radius = 25
    macro = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
    macro[macro < 128] = 0
    macro[macro > 0] = 1

    blur_radius = 5
    mask = cv2.GaussianBlur(mask * macro, (blur_radius, blur_radius), 0)

    # Find bounding box of mask
    nonzero = cv2.findNonZero(mask.astype(np.uint8))
    return mask, cv2.boundingRect(nonzero)
