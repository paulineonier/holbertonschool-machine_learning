#!/usr/bin/env python3
"""
Performs forward propagation over a convolutional layer.
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network.

    Parameters
    ----------
    A_prev : numpy.ndarray
        Shape (m, h_prev, w_prev, c_prev) - output from previous layer.
    W : numpy.ndarray
        Shape (kh, kw, c_prev, c_new) - kernels for the convolution.
    b : numpy.ndarray
        Shape (1, 1, 1, c_new) - biases for the convolution.
    activation : function
        Activation function applied to the convolution output.
    padding : str
        Either 'same' or 'valid', indicating the padding type.
    stride : tuple
        (sh, sw) - strides for the convolution.

    Returns
    -------
    numpy.ndarray
        The output of the convolutional layer after activation.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Determine padding
    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:  # valid
        ph, pw = 0, 0

    # Pad the input
    A_prev_padded = np.pad(
        A_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Output dimensions
    h_new = (h_prev + 2 * ph - kh) // sh + 1
    w_new = (w_prev + 2 * pw - kw) // sw + 1

    # Initialize output
    Z = np.zeros((m, h_new, w_new, c_new))

    # Perform convolution
    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            region = A_prev_padded[:, vert_start:vert_end, horiz_start:horiz_end, :]

            for c in range(c_new):
                kernel = W[:, :, :, c]
                Z[:, i, j, c] = np.sum(region * kernel, axis=(1, 2, 3))

    # Add bias and apply activation
    Z = Z + b
    A = activation(Z)

    return A
