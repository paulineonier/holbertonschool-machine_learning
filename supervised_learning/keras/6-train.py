#!/usr/bin/env python3
"""Train model with optional validation and early stopping"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent, supports validation
    and early stopping.

    Args:
        network: the compiled Keras model to train
        data: input data (numpy.ndarray)
        labels: one-hot labels (numpy.ndarray)
        batch_size: size of mini-batches
        epochs: number of training epochs
        validation_data: tuple (X_valid, Y_valid), optional
        early_stopping: whether to apply early stopping
        patience: number of epochs without improvement before stopping
        verbose: whether to print progress
        shuffle: whether to shuffle data between epochs

    Returns:
        History object from training
    """
    callbacks = []

    if early_stopping and validation_data is not None:
        es_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(es_callback)

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )

    return history
