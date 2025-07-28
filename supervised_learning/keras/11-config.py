#!/usr/bin/env python3
"""Module to save and load the configuration of a Keras model in JSON format"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model’s configuration in JSON format.

    Args:
        network: the model whose configuration should be saved
        filename: path of the file that the configuration should be saved to

    Returns:
        None
    """
    json_config = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_config)


def load_config(filename):
    """
    Loads a model with a specific configuration from JSON.

    Args:
        filename: path of the file containing the model’s configuration

    Returns:
        the loaded model (uncompiled, with no weights)
    """
    with open(filename, 'r') as f:
        json_config = f.read()
    model = K.models.model_from_json(json_config)
    return model
