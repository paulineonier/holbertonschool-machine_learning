#!/usr/bin/env python3
"""Builds a neural network with Keras"""

import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a Keras model with the given specifications.

    Args:
        nx (int): Number of input features.
        layers (list): List of number of nodes in each layer.
        activations (list): List of activation functions for each layer.
        lambtha (float): L2 regularization parameter.
        keep_prob (float): Probability of keeping a node during dropout.

    Returns:
        keras.Model: The compiled Keras model.
    """
    model = K.Sequential()
    reg = K.regularizers.L2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=reg,
                input_shape=(nx,)
            ))
        else:
            model.add(K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=reg
            ))

        # Apply Dropout after each hidden
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(rate=1 - keep_prob))

    return model
