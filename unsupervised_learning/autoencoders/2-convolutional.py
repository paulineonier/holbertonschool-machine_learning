#!/usr/bin/env python3
"""Module that contains the convolutional autoencoder function."""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder model.

    Args:
        input_dims (tuple): Dimensions of the model input (h, w, c).
        filters (list): Number of filters for each convolutional layer in
            the encoder.
        latent_dims (tuple): Dimensions of the latent space representation.

    Returns:
        tuple: (encoder, decoder, auto)
            - encoder: The encoder model.
            - decoder: The decoder model.
            - auto: The full convolutional autoencoder model.
    """
    # --- Encoder ---
    inputs = keras.Input(shape=input_dims)
    x = inputs

    for f in filters:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        )(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    encoder = keras.Model(inputs, x, name='encoder')

    # --- Decoder ---
    latent_inputs = keras.Input(shape=latent_dims)
    x = latent_inputs

    # All decoder convolutions except the last two
    reversed_filters = list(reversed(filters))
    for f in reversed_filters[:-1]:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        )(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Second to last convolution: uses valid padding with upsampling
    x = keras.layers.Conv2D(
        filters=reversed_filters[-1],
        kernel_size=(3, 3),
        activation='relu',
        padding='valid'
    )(x)
    x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Last convolution: output channels, sigmoid activation, no upsampling
    outputs = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        activation='sigmoid',
        padding='same'
    )(x)

    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # --- Autoencoder (Combined) ---
    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs, auto_outputs, name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
