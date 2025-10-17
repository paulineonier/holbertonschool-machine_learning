#!/usr/bin/env python3
"""
Performs a convolution on grayscale images with custom padding.
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

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

    padding : tuple
        Tuple of (ph, pw) representing the padding for height and width.

    Returns
    -------
    numpy.ndarray
        Array containing the convolved images.
        The output shape is (m, h + 2*ph - kh + 1, w + 2*pw - kw + 1).
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad images with zeros
    padded_images = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Compute the shape of the output
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    # Initialize the output array
    conv_output = np.zeros((m, output_h, output_w))

    # Perform convolution using only two loops
    for i in range(output_h):
        for j in range(output_w):
            region = padded_images[:, i:i + kh, j:j + kw]
            conv_output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return conv_output
