#!/usr/bin/env python3
"""
Builds a modified LeNet-5 architecture using Keras.
"""

from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using Keras.

    Parameters
    ----------
    X : K.Input
        Input tensor of shape (m, 28, 28, 1)

    Returns
    -------
    model : K.Model
        The compiled LeNet-5 model
    """
    he_init = K.initializers.HeNormal(seed=0)

    # 1. Convolutional layer (6 filters, 5x5, same padding)
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=he_init
    )(X)

    # 2. Max pooling layer (2x2, stride 2)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # 3. Convolutional layer (16 filters, 5x5, valid padding)
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=he_init
    )(pool1)

    # 4. Max pooling layer (2x2, stride 2)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # 5. Flatten output
    flat = K.layers.Flatten()(pool2)

    # 6. Fully connected layer with 120 nodes
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=he_init
    )(flat)

    # 7. Fully connected layer with 84 nodes
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=he_init
    )(fc1)

    # 8. Output layer with 10 nodes (softmax)
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=he_init
    )(fc2)

    # Build and compile model
    model = K.Model(inputs=X, outputs=output)
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
