#!/usr/bin/env python3
"""Module to save and load only the weights of a Keras model"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves the weights of a Keras model.

    Args:
        network: the model whose weights should be saved
        filename: path of the file to save the weights to
        save_format: format in which to save the weights ('keras' or 'h5')

    Returns:
        None
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    Loads weights into a Keras model.

    Args:
        network: the model to which weights should be loaded
        filename: path of the file from which to load the weights

    Returns:
        None
    """
    network.load_weights(filename)
    return None
