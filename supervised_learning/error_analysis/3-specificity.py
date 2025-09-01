#!/usr/bin/env python3
"""
Module 3-specificity
Contains the function specificity that calculates
the per-class specificity (true negative rate) from a confusion matrix.
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
                TN_i = all samples that are not in row i or column i
                FP_i = sum of column i excluding TP_i
    """
    classes = confusion.shape[0]
    total = np.sum(confusion)
    spec = np.zeros(classes)

    for i in range(classes):
        TP = confusion[i, i]
        FP = np.sum(confusion[:, i]) - TP
        FN = np.sum(confusion[i, :]) - TP
        TN = total - (TP + FP + FN)

        spec[i] = TN / (TN + FP)

    return spec
