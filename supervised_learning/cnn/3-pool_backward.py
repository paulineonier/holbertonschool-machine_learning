#!/usr/bin/env python3
"""
Performs back propagation over a pooling layer of a neural network.
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network.

    Parameters
    ----------
    dA : numpy.ndarray
        (m, h_new, w_new, c_new) partial derivatives w.r.t. pooling output.
    A_prev : numpy.ndarray
        (m, h_prev, w_prev, c_prev) output of previous layer.
    kernel_shape : tuple
        (kh, kw) size of the pooling kernel.
    stride : tuple
        (sh, sw) strides for pooling.
    mode : str
        'max' or 'avg', indicating pooling type.

    Returns
    -------
    dA_prev : numpy.ndarray
        Gradient with respect to the previous layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    _, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Initialize gradient of input
    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            # Backprop for each channel
            if mode == "max":
                a_slice = A_prev[:, vert_start:vert_end,
                                 horiz_start:horiz_end, :]
                mask = (a_slice == np.max(a_slice, axis=(1, 2), keepdims=True))
                dA_prev[:, vert_start:vert_end,
                        horiz_start:horiz_end, :] += mask * dA[:, i:i+1,
                                                               j:j+1, :]
            elif mode == "avg":
                da = dA[:, i:i+1, j:j+1, :]
                average = da / (kh * kw)
                dA_prev[:, vert_start:vert_end,
                        horiz_start:horiz_end, :] += np.ones_like(
                            A_prev[:, vert_start:vert_end,
                                   horiz_start:horiz_end, :]
                        ) * average

    return dA_prev
