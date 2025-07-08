#!/usr/bin/env python3
"""Builds a neural network using the Functional API of Keras"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build a neural network using the Functional API.

    Args:
        nx (int): Number of input features
        layers (list): List of number of nodes per layer
        activations (list): List of activation functions per layer
        lambtha (float): L2 regularization parameter
        keep_prob (float): Dropout keep probability

    Returns:
        keras.Model: The compiled Keras functional model
    """
    # Create input layer
    inputs = K.Input(shape=(nx,))
    reg = K.regularizers.L2(lambtha)

    x = inputs

    for i in range(len(layers)):
        # Add dense layer with L2 regularization and specified activation
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=reg
        )(x)

        # Apply dropout after each hidden layer (not the output)
        if i < len(layers) - 1:
            x = K.layers.Dropout(rate=1 - keep_prob)(x)

    # Create model: input -> output
    model = K.Model(inputs=inputs, outputs=x)
    return model
