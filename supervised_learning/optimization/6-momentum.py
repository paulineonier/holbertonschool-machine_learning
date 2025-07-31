#!/usr/bin/env python3
"""
Creates a momentum optimizer for TensorFlow
"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Creates the Momentum optimization operation for TensorFlow

    Parameters:
    - alpha: learning rate
    - beta1: momentum hyperparameter (also called momentum rate)

    Returns:
    - optimizer: a tf.keras.optimizers.Optimizer instance using momentum
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
