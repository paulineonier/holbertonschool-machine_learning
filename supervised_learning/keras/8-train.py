#!/usr/bin/env python3
"""Trains a model with mini-batch gradient descent,
early stopping, learning rate decay, and best model saving.
"""

import tensorflow.keras as K


def train_model(network, data, labels,
                batch_size, epochs,
                validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False,
                alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    Trains a model with optional early stopping,
    learning rate decay, and best model saving.

    Args:
        network: the Keras model to train
        data: numpy.ndarray, training input data
        labels: numpy.ndarray, one-hot encoded training labels
        batch_size: size of mini-batch
        epochs: number of training epochs
        validation_data: tuple (X_valid, Y_valid)
        early_stopping: enable early stopping
        patience: patience for early stopping
        learning_rate_decay: enable LR decay
        alpha: initial learning rate
        decay_rate: inverse time decay rate
        save_best: save best model based on val_loss
        filepath: path to save the best model
        verbose: whether to print progress
        shuffle: whether to shuffle data per epoch

    Returns:
        History object from training
    """
    callbacks = []

    if validation_data:
        # Learning rate scheduler
        if learning_rate_decay:
            def scheduler(epoch):
                return alpha / (1 + decay_rate * epoch)

            lr_cb = K.callbacks.LearningRateScheduler(scheduler, verbose=1)
            callbacks.append(lr_cb)

        # Early stopping
        if early_stopping:
            es_cb = K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
            callbacks.append(es_cb)

        # Save best model
        if save_best and filepath:
            mc_cb = K.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor='val_loss',
                save_best_only=True
            )
            callbacks.append(mc_cb)

    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)

    return history
