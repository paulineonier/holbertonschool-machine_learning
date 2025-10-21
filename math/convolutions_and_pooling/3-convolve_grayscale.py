#!/usr/bin/env python3
"""
Performs a convolution on grayscale images with flexible padding and stride.
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Parameters
    ----------
    images : numpy.ndarray
        Array of shape (m, h, w) containing multiple grayscale images.
    kernel : numpy.ndarray
        Array of shape (kh, kw) containing the kernel for the convolution.
    padding : tuple or str
        Either a tuple of (ph, pw), 'same', or 'valid'.
    stride : tuple
        Tuple of (sh, sw) strides.

    Returns
    -------
    numpy.ndarray
        Array containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding amounts
    if isinstance(padding, tuple):
        ph, pw = padding
        pad_top, pad_bottom = ph, ph
        pad_left, pad_right = pw, pw
    elif padding == 'same':
        out_h = int(np.ceil(h / sh))
        out_w = int(np.ceil(w / sw))
        pad_h = max((out_h - 1) * sh + kh - h, 0)
        pad_w = max((out_w - 1) * sw + kw - w, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
    else:  # 'valid'
        pad_top = pad_bottom = pad_left = pad_right = 0

    # Pad images
    padded_images = np.pad(
        images,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0
    )

    # Compute output dimensions
    new_h = (padded_images.shape[1] - kh) // sh + 1
    new_w = (padded_images.shape[2] - kw) // sw + 1

    # Initialize output array
    conv_output = np.zeros((m, new_h, new_w), dtype=np.float64)

    # Perform convolution (two loops only)
    for i in range(new_h):
        for j in range(new_w):
            region = padded_images[
                :, i * sh:i * sh + kh, j * sw:j * sw + kw
            ]
            conv_output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return conv_output
