#!/usr/bin/env python3
"""Function to test a Keras model"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network.

    Args:
        network: the trained Keras model to test
        data: the input data to evaluate the model on
        labels: the correct labels (one-hot encoded)
        verbose: if True, print testing progress/output

    Returns:
        Tuple of (loss, accuracy)
    """
    return network.evaluate(data, labels, verbose=verbose)
