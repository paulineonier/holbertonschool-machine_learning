#!/usr/bin/env python3
"""
Performs a convolution on images using multiple kernels.
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Parameters
    ----------
    images : numpy.ndarray
        Array of shape (m, h, w, c) containing multiple images.
    kernels : numpy.ndarray
        Array of shape (kh, kw, c, nc) containing the kernels.
    padding : tuple or str
        Either a tuple of (ph, pw), 'same', or 'valid'.
    stride : tuple
        Tuple of (sh, sw).

    Returns
    -------
    numpy.ndarray
        The convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    # Determine padding
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

    # Apply padding
    padded = np.pad(
        images,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Compute output dimensions
    new_h = (padded.shape[1] - kh) // sh + 1
    new_w = (padded.shape[2] - kw) // sw + 1

    # Initialize output
    output = np.zeros((m, new_h, new_w, nc), dtype=np.float64)

    # Perform convolution (3 loops max)
    for i in range(new_h):
        for j in range(new_w):
            region = padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            for k in range(nc):
                kernel = kernels[:, :, :, k]
                output[:, i, j, k] = np.sum(region * kernel, axis=(1, 2, 3))

    return output
