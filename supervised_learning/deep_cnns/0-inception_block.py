#!/usr/bin/env python3
"""
Builds an inception block as described in
'Going Deeper with Convolutions' (Szegedy et al., 2014)
"""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block.

    Parameters
    ----------
    A_prev : Keras tensor
        Output from the previous layer.

    filters : tuple or list
        Contains (F1, F3R, F3, F5R, F5, FPP):
        - F1:   filters for the 1x1 conv
        - F3R:  filters for the 1x1 conv before the 3x3 conv
        - F3:   filters for the 3x3 conv
        - F5R:  filters for the 1x1 conv before the 5x5 conv
        - F5:   filters for the 5x5 conv
        - FPP:  filters for the 1x1 conv after max pooling

    Returns
    -------
    The concatenated output tensor of the inception block.
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    he_init = K.initializers.HeNormal(seed=0)

    # 1x1 convolution branch
    conv_1x1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=he_init
    )(A_prev)

    # 1x1 followed by 3x3 convolution branch
    conv_3x3_reduce = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=he_init
    )(A_prev)

    conv_3x3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        kernel_initializer=he_init
    )(conv_3x3_reduce)

    # 1x1 followed by 5x5 convolution branch
    conv_5x5_reduce = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=he_init
    )(A_prev)

    conv_5x5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=he_init
    )(conv_5x5_reduce)

    # Max pooling followed by 1x1 convolution branch
    pool_proj = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(A_prev)

    pool_conv = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=he_init
    )(pool_proj)

    # Concatenate all branches
    output = K.layers.Concatenate(axis=-1)(
        [conv_1x1, conv_3x3, conv_5x5, pool_conv]
    )

    return output
