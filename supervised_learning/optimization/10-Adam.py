#!/usr/bin/env python3
"""
Adam optimizer setup in TensorFlow
"""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Sets up the Adam optimization algorithm in TensorFlow.

    Parameters:
    - alpha: learning rate
    - beta1: exponential decay rate for the first moment estimates
    - beta2: exponential decay rate for the second moment estimates
    - epsilon: small constant to prevent division by zero

    Returns:
    - optimizer: a tf.keras.optimizers.Adam instance
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
