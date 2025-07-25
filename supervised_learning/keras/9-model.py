#!/usr/bin/env python3
"""Module to save and load a Keras model"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire Keras model.

    Args:
        network: the Keras model to save
        filename: path of the file to save the model to

    Returns:
        None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Loads an entire Keras model.

    Args:
        filename: path of the file to load the model from

    Returns:
        The loaded Keras model
    """
    return K.models.load_model(filename)
