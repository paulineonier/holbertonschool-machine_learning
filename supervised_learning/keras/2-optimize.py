#!/usr/bin/env python3
"""Function to compile a Keras model with Adam optimizer"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Configures the Adam optimizer for a Keras model.

    Args:
        network: The keras model to compile
        alpha: The learning rate (float)
        beta1: The exponential decay rate for the 1st moment estimates
        beta2: The exponential decay rate for the 2nd moment estimates

    Returns:
        None
    """
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )

    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
