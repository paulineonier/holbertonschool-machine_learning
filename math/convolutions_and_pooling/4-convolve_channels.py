#!/usr/bin/env python3
"""
Performs a convolution on images with multiple channels.
"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Parameters
    ----------
    images : numpy.ndarray
        Array of shape (m, h, w, c) containing multiple images.
        - m: number of images
        - h: height in pixels of each image
        - w: width in pixels of each image
        - c: number of channels in the image

    kernel : numpy.ndarray
        Array of shape (kh, kw, c) containing the kernel for the convolution.
        - kh: height of the kernel
        - kw: width of the kernel
        - c: same number of channels as the images

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
        Array containing the convolved images with shape (m, new_h, new_w).
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if kc != c:
        raise ValueError("Kernel channels must match image channels")

    # Determine padding
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2)
        pw = int(((w - 1) * sw + kw - w) / 2)
    else:  # 'valid'
        ph, pw = 0, 0

    # Pad images with zeros
    padded_images = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Compute output dimensions
    new_h = int(((h + 2 * ph - kh) / sh) + 1)
    new_w = int(((w + 2 * pw - kw) / sw) + 1)

    # Initialize output array
    conv_output = np.zeros((m, new_h, new_w))

    # Perform convolution using only two loops
    for i in range(new_h):
        for j in range(new_w):
            region = padded_images[
                :, i * sh:i * sh + kh, j * sw:j * sw + kw, :
            ]
            conv_output[:, i, j] = np.sum(
                region * kernel,
                axis=(1, 2, 3)
            )

    return conv_output
