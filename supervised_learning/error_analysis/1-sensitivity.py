#!/usr/bin/env python3
"""
Module 1-sensitivity
Computes per-class sensitivity (recall / true positive rate) from a confusion matrix.
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity (recall / true positive rate) for each class.

    Parameters
    confusion : np.ndarray of shape (classes, classes)
        Confusion matrix where rows represent true labels and columns
        represent predicted labels.

    Returns
    np.ndarray of shape (classes,)
        Sensitivity for each class i, defined as:
        sensitivity_i = TP_i / (TP_i + FN_i)
        where:
          - TP_i is the (i, i) entry of the confusion matrix,
          - FN_i is the sum of row i except the diagonal (i.e., missed positives).
    """
    # True positives per class are on the diagonal
    TP = np.diag(confusion)

    # For each class i, the number of actual positives is the sum of row i
    # (TP_i + FN_i)
    actual_positives = np.sum(confusion, axis=1)

    # Vectorized division with safe handling of zero rows:
    # If a class has no actual samples (actual_positives[i] == 0),
    # define its sensitivity as 0.0 to avoid division by zero.
    with np.errstate(divide='ignore', invalid='ignore'):
        sens = np.divide(
            TP,
            actual_positives,
            out=np.zeros_like(TP, dtype=float),
            where=actual_positives != 0
        )

    return sens
