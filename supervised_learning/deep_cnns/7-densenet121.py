#!/usr/bin/env python3
"""
Builds the DenseNet-121 architecture
"""

from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture

    Parameters
    ----------
    growth_rate : int
        Growth rate
    compression : float
        Compression factor

    Returns
    -------
    model : keras.Model
        Keras implementation of DenseNet-121
    """
    he_init = K.initializers.he_normal(seed=0)

    # Input layer
    X = K.Input(shape=(224, 224, 3))

    # Initial Convolution + BN + ReLU + MaxPool
    bn0 = K.layers.BatchNormalization(axis=3)(X)
    act0 = K.layers.Activation('relu')(bn0)
    conv0 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=2,
        padding='same',
        kernel_initializer=he_init
    )(act0)
    pool0 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=2, padding='same')(conv0)

    # Dense Block 1 (6 layers)
    db1, nb_filters = dense_block(pool0, 64, growth_rate, 6)

    # Transition Layer 1
    tl1, nb_filters = transition_layer(db1, nb_filters, compression)

    # Dense Block 2 (12 layers)
    db2, nb_filters = dense_block(tl1, nb_filters, growth_rate, 12)

    # Transition Layer 2
    tl2, nb_filters = transition_layer(db2, nb_filters, compression)

    # Dense Block 3 (24 layers)
    db3, nb_filters = dense_block(tl2, nb_filters, growth_rate, 24)

    # Transition Layer 3
    tl3, nb_filters = transition_layer(db3, nb_filters, compression)

    # Dense Block 4 (16 layers)
    db4, nb_filters = dense_block(tl3, nb_filters, growth_rate, 16)

    # Classification head
    gap = K.layers.AveragePooling2D(pool_size=(7, 7))(db4)
    flat = K.layers.Flatten()(gap)
    output = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=he_init
    )(flat)

    model = K.models.Model(inputs=X, outputs=output)
    return model
