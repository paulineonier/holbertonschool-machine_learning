#!/usr/bin/env python3
"""
6-dropout_create_layer.py
Creates a layer of a neural network with Dropout regularization.
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a new layer with dropout.

    Args:
        prev (tensor): input tensor from the previous layer
        n (int): number of nodes in the new layer
        activation (function): activation function
        keep_prob (float): probability to keep a node active
        training (bool): True if the model is in training mode

    Returns:
        tensor: output of the new layer
        tensor: dropout mask applied
    """
    # Initialize weights using He initialization (shortened)
    W = tf.Variable(
        tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')(
            shape=(prev.shape[-1], n)
        ), dtype=tf.float32
    )
    b = tf.Variable(tf.zeros([n]), dtype=tf.float32)

    # Linear combination and activation
    A = activation(tf.matmul(prev, W) + b)

    # Apply dropout only during training
    if training:
        A = tf.nn.dropout(A, rate=1-keep_prob)
        D = tf.cast(A != 0, tf.float32)
    else:
        D = tf.ones_like(A)

    return A, D
