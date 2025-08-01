#!/usr/bin/env python3
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Parameters:
    - prev: tensor, output of the previous layer
    - n: int, number of nodes in the layer to be created
    - activation: activation function to apply after normalization

    Returns:
    - Tensor of the activated output for the layer
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Dense layer without activation
    dense = tf.keras.layers.Dense(units=n,
                                   kernel_initializer=initializer,
                                   use_bias=False)(prev)

    # Mean and variance for the batch
    mean, variance = tf.nn.moments(dense, axes=[0])

    # Trainable parameters gamma and beta
    gamma = tf.Variable(initial_value=tf.ones([n]), trainable=True)
    beta = tf.Variable(initial_value=tf.zeros([n]), trainable=True)

    # Batch normalization
    epsilon = 1e-7
    batch_norm = tf.nn.batch_normalization(dense, mean, variance, beta, gamma, epsilon)

    # Apply activation function
    return activation(batch_norm)
