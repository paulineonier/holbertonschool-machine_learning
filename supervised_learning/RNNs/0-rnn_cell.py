#!/usr/bin/env python3
"""Module that defines the RNNCell class for a simple RNN cell."""
import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """Initializes the RNN cell.

        Args:
            i (int): Dimensionality of the data input.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        # Concatenated weights for [h_prev, x_t]
        self.Wh = np.random.normal(size=(h + i, h))
        # Weights for output layer
        self.Wy = np.random.normal(size=(h, o))
        # Biases initialized as zeros
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Shape (m, h), previous hidden state.
            x_t (numpy.ndarray): Shape (m, i), input data for the cell.

        Returns:
            tuple: (h_next, y)
                - h_next: Shape (m, h), next hidden state.
                - y: Shape (m, o), output of the cell (softmax).
        """
        # Concatenate previous hidden state and input data horizontally
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Compute next hidden state using tanh activation
        h_next = np.tanh(np.dot(concat_input, self.Wh) + self.bh)

        # Compute unnormalized output scores
        y_linear = np.dot(h_next, self.Wy) + self.by

        # Compute softmax activation for output
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, y
