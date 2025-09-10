#!/usr/bin/env python3
"""
Defines a function to create a layer with L2 regularization
"""

import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a TensorFlow layer that includes L2 regularization.

    Args:
        prev: tensor containing the input to the layer
        n (int): number of nodes in the layer
        activation: activation function to use
        lambtha (float): L2 regularization parameter

    Returns:
        Tensor output of the layer
    """
    # L2 regularizer
    l2_reg = tf.keras.regularizers.L2(lambtha)

    # He et al. initialization (variance scaling initializer)
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg'
    )

    # Dense layer with L2 regularization
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=l2_reg,
        name="layer"
    )

    return layer(prev)
