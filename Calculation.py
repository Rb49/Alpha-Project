# module imports
import numpy as np
import cv2 as cv


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """
    creates a distance map of the mask
    :param mask: binary mask of the fov
    :return: distance map of the mask, with negative values within the mask
    """

    # find edges
    edges = cv.Canny(mask.astype('uint8'), 0, 1)
    edges[edges > 1] = 1

    # invert the edges
    edges = 1 - edges

    dist_transform, labels = cv.distanceTransformWithLabels(edges, cv.DIST_L2, cv.DIST_MASK_PRECISE)

    # make distance negative for places inside the mask
    dist_transform[mask == 1] *= -1

    return dist_transform


def indexByDistance(distance_map: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    converts the centers array value to its proper distance
    :param distance_map: distance map of the mask
    :param centers: index is cell label, [0]: x coord, [1]: y coord
    :return: array where the distance is the cell label, and value is the distance.
    """

    distances = np.zeros((centers.shape[0], 1))
    distances[0] = np.nan

    for i in range(1, centers.shape[0]):
        if round(centers[i, 0]) != -1:
            distances[i] = distance_map[round(centers[i, 0]), round(centers[i, 1])]
        else:  # invalid cell
            distances[i] = np.nan

    return distances
