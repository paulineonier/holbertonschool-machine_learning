#!/usr/bin/env python3
"""
RMSProp Optimizer Setup for TensorFlow
"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Creates the RMSProp optimizer

    Parameters:
    - alpha: learning rate
    - beta2: RMSProp decay(discounting factor for history of squared gradients)
    - epsilon: small constant to avoid division by zero

    Returns:
    - optimizer: a tf.keras.optimizers.RMSprop instance
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
    return optimizer
