#!/usr/bin/env python3
"""
Creates a transition layer as described in DenseNet-C.
Applies:
    - BatchNorm
    - ReLU
    - 1x1 Convolution with compression
    - 2x2 Average Pooling
"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer.

    Parameters
    ----------
    X : tensor
        Output of the previous layer
    nb_filters : int
        Number of filters in X
    compression : float
        Compression factor for DenseNet-C

    Returns
    -------
    Y : tensor
        Output of the transition layer
    new_nb_filters : int
        Number of filters after compression
    """
    he_init = K.initializers.he_normal(seed=0)

    # BatchNorm + ReLU
    bn = K.layers.BatchNormalization(axis=3)(X)
    act = K.layers.Activation('relu')(bn)

    # Number of filters after compression
    new_nb_filters = int(nb_filters * compression)

    # 1x1 Convolution with compression
    conv = K.layers.Conv2D(
        filters=new_nb_filters,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=he_init
    )(act)

    # 2x2 Average Pooling (stride 2)
    pool = K.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=2
    )(conv)

    return pool, new_nb_filters
