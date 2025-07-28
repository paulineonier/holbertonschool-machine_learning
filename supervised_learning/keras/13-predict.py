#!/usr/bin/env python3
"""Function to make a prediction using a Keras model"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network.

    Args:
        network: the Keras model to use for prediction
        data: input data (NumPy array) to make the prediction with
        verbose: if True, prints output during prediction

    Returns:
        The prediction (NumPy array)
    """
    return network.predict(data, verbose=verbose)
