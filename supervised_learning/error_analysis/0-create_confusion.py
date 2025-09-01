#!/usr/bin/env python3
"""
Module 0-create_confusion
Contains the function create_confusion_matrix
that generates a confusion matrix from one-hot encoded labels and predictions.
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from one-hot encoded labels and predictions.

    Args:
        labels (np.ndarray): one-hot array of shape (m, classes)
                             containing the true labels
        logits (np.ndarray): one-hot array of shape (m, classes)
                             containing the predicted labels

    Returns:
        np.ndarray: confusion matrix of shape (classes, classes)
                    (rows = true labels, columns = predicted labels)
    """
    # Convert one-hot encoded arrays to class indices
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)

    # Get the number of classes
    classes = labels.shape[1]

    # Initialize the confusion matrix (filled with zeros)
    confusion = np.zeros((classes, classes))

    # Fill the confusion matrix: increment for each (true, predicted) pair
    for t, p in zip(true_classes, pred_classes):
        confusion[t, p] += 1

    return confusion
