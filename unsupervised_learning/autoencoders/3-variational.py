#!/usr/bin/env python3
"""Module that contains the variational autoencoder function."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a Variational Autoencoder (VAE) model.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in
            the encoder.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        tuple: (encoder, decoder, auto)
            - encoder: The encoder model.
            - decoder: The decoder model.
            - auto: The full VAE model.
    """
    # --- Encoder ---
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_sig = keras.layers.Dense(latent_dims, activation=None)(x)

    # Reparameterization trick / Sampling layer
    def sampling(args):
        """Samples latent vector z from latent distribution."""
        mu, log_sig = args
        epsilon = keras.backend.random_normal(shape=keras.backend.shape(mu))
        return mu + keras.backend.exp(log_sig / 2) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sig])
    encoder = keras.Model(
        inputs,
        [z, z_mean, z_log_sig],
        name='encoder'
    )

    # --- Decoder ---
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # --- Autoencoder (Combined) ---
    z_sampled, mean, log_sig = encoder(inputs)
    auto_outputs = decoder(z_sampled)
    auto = keras.Model(inputs, auto_outputs, name='autoencoder')

    # Reconstruction Loss
    recon_loss = keras.losses.binary_crossentropy(inputs, auto_outputs)
    recon_loss = keras.backend.sum(recon_loss, axis=-1)

    # KL Divergence Loss
    kl_loss = -0.5 * keras.backend.sum(
        1 + log_sig - keras.backend.square(mean) -
        keras.backend.exp(log_sig),
        axis=-1
    )

    vae_loss = keras.backend.mean(recon_loss + kl_loss)
    auto.add_loss(vae_loss)

    auto.compile(optimizer='adam')

    return encoder, decoder, auto
