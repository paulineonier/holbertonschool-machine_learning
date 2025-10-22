#!/usr/bin/env python3
"""
Performs pooling on images.
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Parameters
    ----------
    images : numpy.ndarray
        Array of shape (m, h, w, c) containing multiple images.
    kernel_shape : tuple
        Tuple (kh, kw) for the pooling kernel shape.
    stride : tuple
        Tuple (sh, sw) for the strides.
    mode : str
        Type of pooling, 'max' or 'avg'.

    Returns
    -------
    numpy.ndarray
        Array containing the pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Compute output dimensions
    new_h = (h - kh) // sh + 1
    new_w = (w - kw) // sw + 1

    # Initialize pooled output
    pooled = np.zeros((m, new_h, new_w, c))

    # Perform pooling (two loops only)
    for i in range(new_h):
        for j in range(new_w):
            region = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            if mode == 'max':
                pooled[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(region, axis=(1, 2))

    return pooled
