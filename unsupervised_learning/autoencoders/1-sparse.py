#!/usr/bin/env python3
"""Module that contains the sparse autoencoder function."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates a sparse autoencoder model using L1 regularization.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in
            the encoder.
        latent_dims (int): Dimensions of the latent space representation.
        lambtha (float): Regularization parameter used for L1 regularization
            on the encoded output.

    Returns:
        tuple: (encoder, decoder, auto)
            - encoder: The encoder model.
            - decoder: The decoder model.
            - auto: The full sparse autoencoder model.
    """
    # --- Encoder ---
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Apply L1 activity regularization to the latent layer
    regularizer = keras.regularizers.l1(lambtha)
    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=regularizer
    )(x)
    encoder = keras.Model(inputs, latent, name='encoder')

    # --- Decoder ---
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # --- Autoencoder (Combined) ---
    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs, auto_outputs, name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
