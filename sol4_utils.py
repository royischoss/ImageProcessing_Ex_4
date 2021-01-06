from imageio import imread
from scipy.signal import convolve2d
import numpy as np
from skimage.color import rgb2gray
import scipy.ndimage as spi
from scipy import signal

MAX_SEGMENT = 255
EVERY_TWO = 2
ROWS = 0
COLUMNS = 1
MIN_WIDE = 16
MIN_HEIGHT = 16
END_LEVELS = 0
FIRST = 0
MAX_CLIP = 1
MIN_CLIP = 0


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    The function builds a gaussian pyramid.
    :param im:  a grayscale image with double values in [0, 1].
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter.
    :return: tuple of the resulting pyramid pyr as a standard python array with maximum length of max_levels,
    where each element of the array is a grayscale image and filter_vec which is row vector of shape (1, filter_size)
    used for the pyramid construction.
    """
    pyr = [im]
    conv_vec1 = np.ones((1, 2))
    conv_vec2 = np.ones((1, 2))
    if filter_size == 1:
        filter_vec = np.ones((1, 1))
    else:
        for i in range(filter_size - 2):
            conv_vec1 = signal.convolve(conv_vec1, conv_vec2)
        filter_vec = (1 / np.sum(conv_vec1)) * conv_vec1
    n = im.shape[ROWS]
    m = im.shape[COLUMNS]
    new_image = np.copy(im)
    i = max_levels - 1
    while m > MIN_HEIGHT and n > MIN_WIDE and i > END_LEVELS:
        filtered_image = spi.filters.convolve(spi.filters.convolve(new_image, filter_vec),
                                              filter_vec.T)
        new_image = np.copy(filtered_image[::EVERY_TWO, ::EVERY_TWO])
        pyr.append(new_image)
        i -= 1
        n /= 2
        m /= 2
    return pyr, filter_vec


# from ex1 read image:
def read_image(filename, representation):
    """
    The next lines preform a image read to a matrix of numpy.float64 using
    imagio and numpy libraries.
    :param filename: a path to jpg image we would like to read.
    :param representation: 1 stands for grayscale , 2 for RGB.
    :return: image_mat - a numpy array represents the photo as described above.
    """
    image = imread(filename)
    if representation == 1:
        image_mat = np.array(rgb2gray(image))
    else:
        image_mat = np.array(image.astype(np.float64))
        image_mat /= MAX_SEGMENT
    return image_mat