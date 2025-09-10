#!/usr/bin/env python3
"""
Module 3-l2_reg_create_layer
This module defines a function to create a Keras layer with L2
regularization.

Functions:
    l2_reg_create_layer(prev, n, activation, lambtha):
        Creates a dense layer with L2 regularization.
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in TensorFlow that includes
    L2 regularization.

    Args:
        prev (tf.Tensor): Tensor containing the output of the previous
                          layer.
        n (int): Number of nodes in the new layer.
        activation (callable): Activation function to use in the layer.
        lambtha (float): L2 regularization parameter.

    Returns:
        tf.Tensor: The output tensor of the new layer.
    """
    regularizer = None
    if lambtha > 0:
        regularizer = tf.keras.regularizers.L2(lambtha)

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=regularizer,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0,
            mode='fan_avg'
        )
    )

    return layer(prev)
