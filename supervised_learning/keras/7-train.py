#!/usr/bin/env python3
"""Train model with optional validation, early stopping, and LR decay"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Trains a model with mini-batch gradient descent, validation,
    early stopping, and learning rate decay.

    Args:
        network: compiled keras model
        data: input data (numpy.ndarray)
        labels: one-hot labels (numpy.ndarray)
        batch_size: size of mini-batches
        epochs: number of training epochs
        validation_data: optional tuple (X_valid, Y_valid)
        early_stopping: enable early stopping
        patience: patience for early stopping
        learning_rate_decay: enable inverse time decay on learning rate
        alpha: initial learning rate
        decay_rate: decay rate for learning rate
        verbose: print progress
        shuffle: shuffle data between epochs

    Returns:
        History object from training
    """
    callbacks = []

    # Early stopping
    if early_stopping and validation_data is not None:
        es_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(es_callback)

    # Learning rate decay
    if learning_rate_decay and validation_data is not None:
        def lr_schedule(epoch):
            new_lr = alpha / (1 + decay_rate * epoch)
            print(f"\nEpoch {epoch + 1}: Learning rate is {new_lr:.6f}")
            return new_lr

        lr_callback = K.callbacks.LearningRateScheduler(
            lr_schedule,
            verbose=1
        )
        callbacks.append(lr_callback)

    # Training the model
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
