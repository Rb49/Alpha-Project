# module imports
from skimage import io
import numpy as np
import pandas as pd
import cv2 as cv


def remove_noise(name: str) -> np.ndarray:
    """
    function that removes the noise from the fov
    :param name: name of fov
    :return: returns an array of the cleared fov
    """

    im = io.imread(str("C:\\Users\\roeyb\\PycharmProjects\\Alpha\\data\\masks\\" + name + "_pixel_mask.tiff"))
    # plt.figure(); plt.imshow(im)

    # masking only epithel clusters
    im_masked = np.zeros_like(im)
    # epithel = 2, 3 mucin = 6
    im_masked[pd.DataFrame(im).isin([2, 3, 6])] = 1
    # plt.imshow(im_masked)

    # median filter
    im_median = cv.medianBlur(im_masked.astype('uint8'),
                              21)  # Add median filter to image
    im_median = cv.medianBlur(im_median.astype('uint8'),
                              19)  # Add median filter to image
    im_median = cv.medianBlur(im_median.astype('uint8'),
                              17)  # Add median filter to image
    im_median = cv.medianBlur(im_median.astype('uint8'),
                              15)  # Add median filter to image

    # plt.imshow(im_median)

    # mask_blurred = im_median
    # blurring
    mask_blurred = cv.blur(im_median.astype(float), (3, 3))
    # debug
    # mask_blurred = cv2.blur(mask_blurred.astype(float) , (3 ,3))
    # mask_blurred = cv2.GaussianBlur(im_masked,(5,5),cv2.BORDER_DEFAULT)
    mask_blurred[mask_blurred >= 1 / 25] = 1
    mask_blurred[mask_blurred < 1 / 25] = 0
    # plt.imshow(mask_blurred)

    im = mask_blurred
    # plt.figure(); plt.imshow(im)

    return im


def read_cell_segments(segmentsPath: str) -> np.ndarray:
    """
    function that reads cell segmentation tiff and calculates cells middle points
    :param segmentsPath: directory path of segments .tiff file
    :return: returns np.ndarray of fov mask with labeled pixels (cells middles)
    """

    # read cell segmentation
    segments = io.imread(segmentsPath)

    unique_size = np.amax(np.unique(segments)) + 1

    # mean counters
    sigmaX = np.zeros(unique_size)
    sigmaY = np.zeros(unique_size)
    n_counter = np.zeros(unique_size)

    # middle points array
    centers = np.zeros([unique_size, 2])
    # fill counters
    for x in range(0, segments.shape[0]):
        for y in range(0, segments.shape[1]):
            value = segments[x, y]
            sigmaX[value] += x
            sigmaY[value] += y
            n_counter[value] += 1
    # calculate middle points and put them is 'centers' (exclude 0 and 1 for edges and background)
    for i in range(1, unique_size):
        if n_counter[i] != 0:
            centers[i, 0] = round(sigmaX[i] / n_counter[i])
            centers[i, 1] = round(sigmaY[i] / n_counter[i])
        else:  # invalid cell
            centers[i, :] = [-1, -1]

    # debug
    # for i in range(0, centers.shape[1]):
    #     print(centers[0, i], centers[1, i])

    return centers
