#!/usr/bin/env python3
"""
Performs forward propagation over a pooling layer of a neural network.
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.

    Parameters
    ----------
    A_prev : numpy.ndarray
        Shape (m, h_prev, w_prev, c_prev), output from previous layer.
    kernel_shape : tuple
        (kh, kw), size of the kernel for pooling.
    stride : tuple
        (sh, sw), strides for the pooling.
    mode : str
        Either 'max' or 'avg', indicating pooling type.

    Returns
    -------
    numpy.ndarray
        The output of the pooling layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Compute output dimensions
    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    # Initialize the output
    A = np.zeros((m, h_new, w_new, c_prev))

    # Perform pooling
    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            region = A_prev[:, vert_start:vert_end, horiz_start:horiz_end, :]

            if mode == 'max':
                A[:, i, j, :] = np.max(region, axis=(1, 2))
            else:  # 'avg'
                A[:, i, j, :] = np.mean(region, axis=(1, 2))

    return A
