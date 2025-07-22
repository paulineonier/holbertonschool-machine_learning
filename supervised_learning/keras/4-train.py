#!/usr/bin/env python3
"""trains a model using mini-batch gradient descent"""

import tensorflow.keras as K


def train_model(network, data, labels,
                batch_size, epochs,
                verbose=True, shuffle=False):
    """
    trains a model using mini-batch gradient descent

    Args :
        network: model to train
        data np(m, nx): numpy array containing label of data
        labels: one-hot numpy array
        batch_size: size of the batch
        epochs: umber of passes through data for mini batch gradient descent
        verbose:

    Return:
        the History object generated after training the model

    """

    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
