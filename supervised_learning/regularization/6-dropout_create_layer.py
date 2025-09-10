#!/usr/bin/env python3
"""
6-dropout_create_layer.py
Creates a layer of a neural network with Dropout regul' in TensorFlow 2.x
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a neural network layer using Dropout.

    Args:
        prev: tensor, output of the previous layer
        n (int): number of nodes for the new layer
        activation: activation function for the layer
        keep_prob (float): probability that a node will be kept
        training (bool): whether the model is in training mode

    Returns:
        Tuple of (output tensor, dropout mask tensor)
    """
    # He initialization for better convergence
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg'
    )

    # Fully connected dense layer
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )

    # Apply the layer to the previous output
    A = layer(prev)

    # Apply Dropout only in training mode
    if training and keep_prob < 1.0:
        # dropout layer in TF2 returns the scaled output
        D = tf.nn.dropout(A, rate=1 - keep_prob)
    else:
        D = A

    return D, D  # Return output and mask (for compatibility)
