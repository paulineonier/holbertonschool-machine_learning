#!/usr/bin/env python3
"""
Identity block for ResNet (2015)
"""
import tensorflow as tf
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in
    'Deep Residual Learning for Image Recognition' (2015)

    Args:
        A_prev: output from the previous layer
        filters: tuple/list of (F11, F3, F12)

    Returns:
        Activated output of the identity block
    """

    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    # First 1x1 convolution
    X = K.layers.Conv2D(F11, (1, 1),
                        padding='same',
                        kernel_initializer=initializer)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # 3x3 convolution
    X = K.layers.Conv2D(F3, (3, 3),
                        padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second 1x1 convolution
    X = K.layers.Conv2D(F12, (1, 1),
                        padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Add shortcut
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
