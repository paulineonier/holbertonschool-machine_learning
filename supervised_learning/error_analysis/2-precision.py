#!/usr/bin/env python3
"""
Module 2-precision
Contains the function precision that calculates
the per-class precision from a confusion matrix.
"""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Args:
        confusion (np.ndarray): shape (classes, classes)
            Confusion matrix where rows = true labels,
            and columns = predicted labels.

    Returns:
        np.ndarray of shape (classes,)
            Precision for each class i:
                precision_i = TP_i / (TP_i + FP_i)
            where:
                TP_i = confusion[i, i]
                FP_i = sum(confusion[:, i]) - TP_i
    """
    # True positives on the diagonal
    TP = np.diag(confusion)

    # Predicted positives per class = sum of column (TP + FP)
    predicted_positives = np.sum(confusion, axis=0)

    # Precision = TP / (TP + FP)
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.divide(
            TP,
            predicted_positives,
            out=np.zeros_like(TP, dtype=float),
            where=predicted_positives != 0
        )

    return precision
