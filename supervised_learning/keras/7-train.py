#!/usr/bin/env python3
"""Trains a model using mini-batch gradient descent with
optional early stopping and learning rate decay.
"""

import tensorflow.keras as K


def train_model(network, data, labels,
                batch_size, epochs,
                validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False,
                alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent

    Args:
        network: model to train
        data: input data (numpy.ndarray) shape (m, nx)
        labels: one-hot labels (numpy.ndarray) shape (m, classes)
        batch_size: size of mini-batch
        epochs: number of passes through data
        validation_data: tuple of validation data (X_valid, Y_valid)
        early_stopping: whether to use early stopping
        patience: patience for early stopping
        learning_rate_decay: whether to use inverse time decay
        alpha: initial learning rate
        decay_rate: decay rate
        verbose: display verbose output
        shuffle: shuffle data between epochs

    Returns:
        History object from training
    """
    callbacks = []

    # Learning rate decay (only if validation_data is provided)
    if validation_data and learning_rate_decay:
        def lr_schedule(epoch):
            return alpha / (1 + decay_rate * epoch)

        lr_callback = K.callbacks.LearningRateScheduler(lr_schedule,
                                                        verbose=1)
        callbacks.append(lr_callback)

    # Early stopping (only if validation_data is provided)
    if validation_data and early_stopping:
        es_callback = K.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience)
        callbacks.append(es_callback)

    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)

    return history
