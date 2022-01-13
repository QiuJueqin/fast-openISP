# File: helpers.py
# Description: Numpy helpers for image processing
# Created: 2021/10/22 20:34
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np


def get_bayer_indices(pattern):
    """
    Get (x_start_idx, y_start_idx) for R, Gr, Gb, and B channels
    in Bayer array, respectively
    """
    return {'gbrg': ((0, 1), (1, 1), (0, 0), (1, 0)),
            'rggb': ((0, 0), (1, 0), (0, 1), (1, 1)),
            'bggr': ((1, 1), (0, 1), (1, 0), (0, 0)),
            'grbg': ((1, 0), (0, 0), (1, 1), (0, 1))}[pattern.lower()]


def split_bayer(bayer_array, bayer_pattern):
    """
    Split R, Gr, Gb, and B channels sub-array from a Bayer array
    :param bayer_array: np.ndarray(H, W)
    :param bayer_pattern: 'gbrg' | 'rggb' | 'bggr' | 'grbg'
    :return: 4-element list of R, Gr, Gb, and B channel sub-arrays, each is an np.ndarray(H/2, W/2)
    """
    rggb_indices = get_bayer_indices(bayer_pattern)

    sub_arrays = []
    for idx in rggb_indices:
        x0, y0 = idx
        sub_arrays.append(
            bayer_array[y0::2, x0::2]
        )

    return sub_arrays


def reconstruct_bayer(sub_arrays, bayer_pattern):
    """
    Inverse implementation of split_bayer: reconstruct a Bayer array from a list of
        R, Gr, Gb, and B channel sub-arrays
    :param sub_arrays: 4-element list of R, Gr, Gb, and B channel sub-arrays, each np.ndarray(H/2, W/2)
    :param bayer_pattern: 'gbrg' | 'rggb' | 'bggr' | 'grbg'
    :return: np.ndarray(H, W)
    """
    rggb_indices = get_bayer_indices(bayer_pattern)

    height, width = sub_arrays[0].shape
    bayer_array = np.empty(shape=(2 * height, 2 * width), dtype=sub_arrays[0].dtype)

    for idx, sub_array in zip(rggb_indices, sub_arrays):
        x0, y0 = idx
        bayer_array[y0::2, x0::2] = sub_array

    return bayer_array


def pad(array, pads, mode='reflect'):
    """
    Pad an array with given margins
    :param array: np.ndarray(H, W, ...)
    :param pads: {int, sequence}
        if int, pad top, bottom, left, and right directions with the same margin
        if 2-element sequence: (y-direction pad, x-direction pad)
        if 4-element sequence: (top pad, bottom pad, left pad, right pad)
    :param mode: padding mode, see np.pad
    :return: padded array: np.ndarray(H', W', ...)
    """
    if isinstance(pads, (list, tuple, np.ndarray)):
        if len(pads) == 2:
            pads = ((pads[0], pads[0]), (pads[1], pads[1])) + ((0, 0),) * (array.ndim - 2)
        elif len(pads) == 4:
            pads = ((pads[0], pads[1]), (pads[2], pads[3])) + ((0, 0),) * (array.ndim - 2)
        else:
            raise NotImplementedError

    return np.pad(array, pads, mode)


def crop(array, crops):
    """
    Crop an array by given margins
    :param array: np.ndarray(H, W, ...)
    :param crops: {int, sequence}
        if int, crops top, bottom, left, and right directions with the same margin
        if 2-element sequence: (y-direction crop, x-direction crop)
        if 4-element sequence: (top crop, bottom crop, left crop, right crop)
    :return: cropped array: np.ndarray(H', W', ...)
    """
    if isinstance(crops, (list, tuple, np.ndarray)):
        if len(crops) == 2:
            top_crop = bottom_crop = crops[0]
            left_crop = right_crop = crops[1]
        elif len(crops) == 4:
            top_crop, bottom_crop, left_crop, right_crop = crops
        else:
            raise NotImplementedError
    else:
        top_crop = bottom_crop = left_crop = right_crop = crops

    height, width = array.shape[:2]
    return array[top_crop: height - bottom_crop, left_crop: width - right_crop, ...]


def shift_array(padded_array, window_size):
    """
    Shift an array within a window and generate window_size**2 shifted arrays
    :param padded_array: np.ndarray(H+2r, W+2r)
    :param window_size: 2r+1
    :return: a generator of length (2r+1)*(2r+1), each is an np.ndarray(H, W), and the original
        array before padding locates in the middle of the generator
    """
    wy, wx = window_size if isinstance(window_size, (list, tuple)) else (window_size, window_size)
    assert wy % 2 == 1 and wx % 2 == 1, 'only odd window size is valid'

    height = padded_array.shape[0] - wy + 1
    width = padded_array.shape[1] - wx + 1

    for y0 in range(wy):
        for x0 in range(wx):
            yield padded_array[y0:y0 + height, x0:x0 + width, ...]


def gen_gaussian_kernel(kernel_size, sigma):
    if isinstance(kernel_size, (list, tuple)):
        assert len(kernel_size) == 2
        wy, wx = kernel_size
    else:
        wy = wx = kernel_size

    x = np.arange(wx) - wx // 2
    if wx % 2 == 0:
        x += 0.5

    y = np.arange(wy) - wy // 2
    if wy % 2 == 0:
        y += 0.5

    y, x = np.meshgrid(y, x)

    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def generic_filter(array, kernel):
    """
    Filter input image array with given kernel
    :param array: array to be filter: np.ndarray(H, W, ...), must be np.int dtype
    :param kernel: np.ndarray(h, w)
    :return: filtered array: np.ndarray(H, W, ...)
    """
    kh, kw = kernel.shape[:2]
    kernel = kernel.flatten()

    padded_array = pad(array, pads=(kh // 2, kw // 2))
    shifted_arrays = shift_array(padded_array, window_size=(kh, kw))

    filtered_array = np.zeros_like(array)
    weights = np.zeros_like(array)

    for i, shifted_array in enumerate(shifted_arrays):
        filtered_array += kernel[i] * shifted_array
        weights += kernel[i]

    filtered_array = (filtered_array / weights).astype(array.dtype)
    return filtered_array


def mean_filter(array, filter_size=3):
    """
    A faster reimplementation of the mean filter
    :param array: array to be filter: np.ndarray(H, W, ...)
    :param filter_size: int, diameter of the mean-filter
    :return: filtered array: np.ndarray(H, W, ...)
    """

    assert filter_size % 2 == 1, 'only odd filter size is valid'

    padded_array = pad(array, pads=filter_size // 2)
    shifted_arrays = shift_array(padded_array, window_size=filter_size)
    return (sum(shifted_arrays) / filter_size ** 2).astype(array.dtype)


def bilateral_filter(array, spatial_weights, intensity_weights_lut, right_shift=0):
    """
    A faster reimplementation of the bilateral filter
    :param array: array to be filter: np.ndarray(H, W, ...), must be np.int dtype
    :param spatial_weights: np.ndarray(h, w): predefined spatial gaussian kernel, where h and w are
        kernel height and width respectively
    :param intensity_weights_lut: a predefined exponential LUT that maps intensity distance to the weight
    :param right_shift: shift the multiplication result of the spatial- and intensity weights to the
        right to avoid integer overflow when multiply this result to the input array
    :return: filtered array: np.ndarray(H, W, ...)
    """
    filter_height, filter_width = spatial_weights.shape[:2]
    spatial_weights = spatial_weights.flatten()

    padded_array = pad(array, pads=(filter_height // 2, filter_width // 2))
    shifted_arrays = shift_array(padded_array, window_size=(filter_height, filter_width))

    bf_array = np.zeros_like(array)
    weights = np.zeros_like(array)

    for i, shifted_array in enumerate(shifted_arrays):
        intensity_diff = (shifted_array - array) ** 2
        weight = intensity_weights_lut[intensity_diff] * spatial_weights[i]
        weight = np.right_shift(weight, right_shift)  # to avoid overflow

        bf_array += weight * shifted_array
        weights += weight

    bf_array = (bf_array / weights).astype(array.dtype)

    return bf_array
