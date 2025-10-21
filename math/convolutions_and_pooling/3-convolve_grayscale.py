#!/usr/bin/env python3
"""
Performs a convolution on grayscale images with custom padding and stride.
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

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

    padding : tuple or str
        Either a tuple of (ph, pw), 'same', or 'valid'.
        - 'same' : output has same height and width as input.
        - 'valid': no padding.
        - tuple  : (ph, pw) custom padding.

    stride : tuple
        Tuple of (sh, sw):
        - sh: stride along height
        - sw: stride along width

    Returns
    -------
    numpy.ndarray
        Array containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2)
        pw = int(((w - 1) * sw + kw - w) / 2)
    else:  # 'valid'
        ph, pw = 0, 0

    # Pad the images with zeros
    padded_images = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Compute output dimensions
    ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
    pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))


    # Initialize output array
    conv_output = np.zeros((m, output_h, output_w))

    # Perform convolution using only two loops
    for i in range(output_h):
        for j in range(output_w):
            region = padded_images[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            conv_output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return conv_output
