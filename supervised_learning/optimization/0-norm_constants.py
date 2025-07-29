#!/usr/bin/env python3
"""Module that calculates normalization constants using TensorFlow"""

import tensorflow as tf


def normalization_constants(X):
    """
    Calculates the mean and standard deviation of a dataset using TensorFlow.

    Args:
        X (tf.Tensor): Tensor of shape (m, nx) representing the data

    Returns:
        tuple: (mean, stddev) - Tensors of shape (nx,)
    """
    mean = tf.math.reduce_mean(X, axis=0)
    stddev = tf.math.reduce_std(X, axis=0)
    return mean, stddev
