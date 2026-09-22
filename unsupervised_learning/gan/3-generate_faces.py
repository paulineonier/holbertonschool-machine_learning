#!/usr/bin/env python3
"""
Module 3-generate_faces
Contains the convolutional_GenDiscr function that builds a convolutional
Generator and Discriminator for face generation.
"""

from tensorflow import keras


def convolutional_GenDiscr():
    """
    Builds a convolutional Generator and Discriminator for face generation.

    Returns:
        tuple: A tuple (generator, discriminator) containing the compiled
               Keras models.
    """

    def get_generator():
        """
        Constructs the Generator model.

        Returns:
            keras.Model: Generator network.
        """
        inputs = keras.Input(shape=(16,))
        x = keras.layers.Dense(2048, activation='tanh')(inputs)
        x = keras.layers.Reshape((2, 2, 512))(x)

        # Block 1 (2x2 -> 4x4)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)

        # Block 2 (4x4 -> 8x8)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(16, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)

        # Block 3 (8x8 -> 16x16)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(1, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Activation('tanh')(x)

        return keras.Model(inputs, outputs, name="generator")

    def get_discriminator():
        """
        Constructs the Discriminator model.

        Returns:
            keras.Model: Discriminator network.
        """
        inputs = keras.Input(shape=(16, 16, 1))

        # Block 1 (16x16 -> 8x8)
        x = keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Activation('tanh')(x)

        # Block 2 (8x8 -> 4x4)
        x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Activation('tanh')(x)

        # Block 3 (4x4 -> 2x2)
        x = keras.layers.Conv2D(128, (3, 3), padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Activation('tanh')(x)

        # Block 4 (2x2 -> 1x1)
        x = keras.layers.Conv2D(256, (3, 3), padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Activation('tanh')(x)

        # Output
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1, activation='tanh')(x)

        return keras.Model(inputs, outputs, name="discriminator")

    return get_generator(), get_discriminator()
