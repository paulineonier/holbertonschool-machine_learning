#!/usr/bin/env python3
"""Module that defines the LSTMCell class for a Long Short-Term Memory unit."""
import numpy as np


class LSTMCell:
    """Represents a Long Short-Term Memory (LSTM) cell."""

    def __init__(self, i, h, o):
        """Initializes the LSTM cell.

        Args:
            i (int): Dimensionality of the data input.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        # Forget gate weights and biases
        self.Wf = np.random.normal(size=(h + i, h))
        self.bf = np.zeros((1, h))

        # Update (input) gate weights and biases
        self.Wu = np.random.normal(size=(h + i, h))
        self.bu = np.zeros((1, h))

        # Intermediate cell state candidate weights and biases
        self.Wc = np.random.normal(size=(h + i, h))
        self.bc = np.zeros((1, h))

        # Output gate weights and biases
        self.Wo = np.random.normal(size=(h + i, h))
        self.bo = np.zeros((1, h))

        # Output projection weights and biases
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Shape (m, h), previous hidden state.
            c_prev (numpy.ndarray): Shape (m, h), previous cell state.
            x_t (numpy.ndarray): Shape (m, i), input data for the cell.

        Returns:
            tuple: (h_next, c_next, y)
                - h_next: Shape (m, h), next hidden state.
                - c_next: Shape (m, h), next cell state.
                - y: Shape (m, o), output of the cell (softmax).
        """
        # Concatenate previous hidden state and input data horizontally
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # 1. Forget gate (f_t)
        f_t = 1 / (1 + np.exp(-(np.dot(concat_input, self.Wf) + self.bf)))

        # 2. Update/Input gate (u_t)
        u_t = 1 / (1 + np.exp(-(np.dot(concat_input, self.Wu) + self.bu)))

        # 3. Intermediate cell state candidate (c_tilde)
        c_tilde = np.tanh(np.dot(concat_input, self.Wc) + self.bc)

        # 4. Next cell state (c_next)
        c_next = f_t * c_prev + u_t * c_tilde

        # 5. Output gate (o_t)
        o_t = 1 / (1 + np.exp(-(np.dot(concat_input, self.Wo) + self.bo)))

        # 6. Next hidden state (h_next)
        h_next = o_t * np.tanh(c_next)

        # 7. Output calculation with Softmax
        y_linear = np.dot(h_next, self.Wy) + self.by
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, c_next, y
