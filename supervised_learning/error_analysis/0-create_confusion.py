#!/usr/bin/env python3
import numpy as np

def create_confusion_matrix(labels, logits):
    """
    Crée une matrice de confusion à partir de labels et de prédictions one-hot.

    Args:
        labels (np.ndarray): matrice one-hot de forme (m, classes) avec les vraies étiquettes
        logits (np.ndarray): matrice one-hot de forme (m, classes) avec les prédictions

    Returns:
        np.ndarray: matrice de confusion de forme (classes, classes)
                    (lignes = vrais labels, colonnes = prédictions)
    """
    # Conversion one-hot -> indices
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)

    # Nombre de classes
    classes = labels.shape[1]

    # Initialisation de la matrice
    confusion = np.zeros((classes, classes))

    # Remplissage
    for t, p in zip(true_classes, pred_classes):
        confusion[t, p] += 1

    return confusion
