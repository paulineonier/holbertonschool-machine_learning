#!/usr/bin/env python3
"""
Performs a same convolution on grayscale images.
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Parameters
    ----------
    images : numpy.ndarray
        Array of shape (m, h, w) containing multiple grayscale images.
        - m: number of images
        - h: height in pixels of each image
        - w: width in pixels of each image

    kernel : numpy.ndarray
        Array of shape (kh, kw) containing the kernel for the convolution.
        - kh: height of the kernel
        - kw: width of the kernel

    Returns
    -------
    numpy.ndarray
        Array containing the convolved images with shape (m, h, w).
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Compute the padding for 'same' convolution
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad images with zeros
    padded_images = np.pad(
        images,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant'
    )

    # Initialize the output array
    conv_output = np.zeros((m, h, w))

    # Perform convolution with only two loops
    for i in range(h):
        for j in range(w):
            region = padded_images[:, i:i + kh, j:j + kw]
            conv_output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return conv_output
