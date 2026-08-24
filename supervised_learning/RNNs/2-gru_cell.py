#!/usr/bin/env python3
"""Module that defines the GRUCell class for a Gated Recurrent Unit."""
import numpy as np


class GRUCell:
    """Represents a Gated Recurrent Unit (GRU) cell."""

    def __init__(self, i, h, o):
        """Initializes the GRU cell.

        Args:
            i (int): Dimensionality of the data input.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        # Update gate weights and biases
        self.Wz = np.random.normal(size=(h + i, h))
        self.bz = np.zeros((1, h))

        # Reset gate weights and biases
        self.Wr = np.random.normal(size=(h + i, h))
        self.br = np.zeros((1, h))

        # Intermediate hidden state weights and biases
        self.Wh = np.random.normal(size=(h + i, h))
        self.bh = np.zeros((1, h))

        # Output weights and biases
        self.Wy = np.random.normal(size=(h, o))
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
        # Concatenate previous hidden state and input data
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # 1. Update gate (z_t) with sigmoid
        z_t = 1 / (1 + np.exp(-(np.dot(concat_input, self.Wz) + self.bz)))

        # 2. Reset gate (r_t) with sigmoid
        r_t = 1 / (1 + np.exp(-(np.dot(concat_input, self.Wr) + self.br)))

        # 3. Intermediate hidden state candidate (h_tilde)
        concat_reset = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.dot(concat_reset, self.Wh) + self.bh)

        # 4. Next hidden state (h_next)
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # 5. Output calculation with Softmax
        y_linear = np.dot(h_next, self.Wy) + self.by
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, y
