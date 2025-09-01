#!/usr/bin/env python3
"""
Module 3-specificity
Contains the function specificity that calculates
the per-class specificity (true negative rate) from confusion matrix.
"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity (true negative rate) for each class.

    Args:
        confusion (np.ndarray): shape (classes, classes)
            Confusion matrix where rows = true labels,
            columns = predicted labels.

    Returns:
        np.ndarray of shape (classes,)
            Specificity of each class i:
                specificity_i = TN_i / (TN_i + FP_i)
            where:
                TN_i = sum of all elements not in row i or column i
                FP_i = sum of column i excluding the diagonal
    """
    classes = confusion.shape[0]
    total = np.sum(confusion)  # Total number of examples
    TN = np.zeros(classes)     # True negatives for each class

    for i in range(classes):
        TP = confusion[i, i]
        FP = np.sum(confusion[:, i]) - TP
        FN = np.sum(confusion[i, :]) - TP
        # True negatives = all other entries not in row i or column i
        TN[i] = total - (TP + FP + FN)

    # Specificity = TN / (TN + FP)
    with np.errstate(divide='ignore', invalid='ignore'):
        specificity = np.divide(
            TN,
            TN + FP,
            out=np.zeros_like(TN, dtype=float),
            where=(TN + FP) != 0
        )

    return specificity
