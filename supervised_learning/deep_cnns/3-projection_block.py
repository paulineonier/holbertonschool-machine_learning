#!/usr/bin/env python3
"""
Projection block for ResNet (2015)
"""
import tensorflow as tf
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in
    'Deep Residual Learning for Image Recognition' (2015)

    Args:
        A_prev: output from previous layer
        filters: tuple/list (F11, F3, F12)
        s: stride for the first conv layers

    Returns:
        Activated output of the projection block
    """

    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    # MAIN PATH
    # First 1x1 conv with stride s
    X = K.layers.Conv2D(F11, (1, 1),
                        strides=s,
                        padding='same',
                        kernel_initializer=initializer)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # 3x3 conv
    X = K.layers.Conv2D(F3, (3, 3),
                        padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Final 1x1 conv
    X = K.layers.Conv2D(F12, (1, 1),
                        padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # SHORTCUT PATH
    shortcut = K.layers.Conv2D(F12, (1, 1),
                               strides=s,
                               padding='same',
                               kernel_initializer=initializer)(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # ADD + RELU
    X = K.layers.Add()([X, shortcut])
    X = K.layers.Activation('relu')(X)

    return X
