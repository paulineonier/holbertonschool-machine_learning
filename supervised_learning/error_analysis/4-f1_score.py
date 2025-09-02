#!/usr/bin/env python3
"""
Module 4-f1_score
Contains the function f1_score that calculates
the F1 score for each class from a confusion matrix.
"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class.

    Args:
        confusion (np.ndarray): shape (classes, classes)
            Confusion matrix where rows = true labels,
            columns = predicted labels.

    Returns:
        np.ndarray of shape (classes,)
            F1 score of each class i.
    """
    # Compute sensitivity (recall) and precision per class
    sens = sensitivity(confusion)
    prec = precision(confusion)

    #  F1 formule: 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (prec * sens) / (prec + sens)

    return f1
