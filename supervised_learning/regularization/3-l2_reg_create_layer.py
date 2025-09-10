#!/usr/bin/env python3
"""
3-l2_reg_create_layer.py
Defines a function to create a layer with L2 regularization in TensorFlow 2.x
"""

import tensorflow as tf


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
    # Define L2 regularizer with parameter lambtha
    l2_reg = tf.keras.regularizers.L2(l2=lambtha)

    # Use He initialization (VarianceScaling) for better convergence
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg'
    )

    # Create a Dense layer with L2 regularization
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=l2_reg
    )

    # Apply the layer to the previous output
    return layer(prev)
