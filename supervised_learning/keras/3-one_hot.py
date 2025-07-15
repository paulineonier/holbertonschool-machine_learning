#!/usr/bin/env python3
"""Converts a label vector into a one-hot matrix using TensorFlow"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Args:
        labels (tensor): Tensor of integer class labels
        classes (int): Total number of classes. If None, use max label + 1.

    Returns:
        Tensor: One-hot encoded matrix (TensorFlow tensor)
    """
    if classes is None:
        classes = K.backend.max(labels) + 1

    return K.utils.to_categorical(labels, num_classes=classes)
