#!/usr/bin/env python3
"""Trains a model using mini-batch gradient descent"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent

    Args:
        network: The Keras model to train
        data: numpy.ndarray of shape (m, nx) with the input data
        labels: one-hot numpy.ndarray of shape (m, classes) with labels
        batch_size: batch size for mini-batch gradient descent
        epochs: number of passes through the data
        validation_data: data for validation (optional)
        verbose: whether to output progress during training
        shuffle: whether to shuffle the data every epoch

    Returns:
        History object generated after training the model
    """
    return network.fit(data, labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data)
