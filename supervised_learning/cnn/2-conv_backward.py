#!/usr/bin/env python3
"""
Performs back propagation over a convolutional layer of a neural network.
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.

    Parameters
    ----------
    dZ : numpy.ndarray
        (m, h_new, w_new, c_new) - partial derivatives w.r.t.
    A_prev : numpy.ndarray
        (m, h_prev, w_prev, c_prev) - output of the previous layer
    W : numpy.ndarray
        (kh, kw, c_prev, c_new) - kernels
    b : numpy.ndarray
        (1, 1, 1, c_new) - biases
    padding : str
        'same' or 'valid'
    stride : tuple
        (sh, sw)

    Returns
    -------
    dA_prev, dW, db : tuple of numpy.ndarray
        Gradients with respect to previous layer, kernels, and biases
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Initialize gradients
    dA_prev = np.zeros((m, h_prev, w_prev, c_prev))
    dW = np.zeros((kh, kw, c_prev, c_new))
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Determine padding
    if padding == 'same':
        pad_h = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pad_w = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    else:
        pad_h, pad_w = 0, 0

    # Pad A_prev and dA_prev
    A_prev_pad = np.pad(A_prev,
                        ((0, 0), (pad_h, pad_h),
                         (pad_w, pad_w), (0, 0)),
                        mode='constant', constant_values=0)
    dA_prev_pad = np.pad(dA_prev,
                         ((0, 0), (pad_h, pad_h),
                          (pad_w, pad_w), (0, 0)),
                         mode='constant', constant_values=0)

    # Loop over images
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(h_new):
            for w in range(w_new):
                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw

                # Slice input
                a_slice = a_prev_pad[vert_start:vert_end,
                                     horiz_start:horiz_end, :]

                # Compute gradients
                for c in range(c_new):
                    da_prev_pad[vert_start:vert_end,
                                horiz_start:horiz_end, :] += (
                        W[:, :, :, c] * dZ[i, h, w, c]
                    )
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
        # Store unpadded gradient
        if pad_h > 0 or pad_w > 0:
            dA_prev[i, :, :, :] = da_prev_pad[pad_h:-pad_h or None,
                                              pad_w:-pad_w or None, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad

    return dA_prev, dW, db
