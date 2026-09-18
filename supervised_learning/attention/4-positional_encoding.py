#!/usr/bin/env python3
"""Module pour le calcul de l'encodage positionnel dans un Transformer."""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calcule l'encodage positionnel pour un Transformer.

    Parameters:
    max_seq_len (int): la longueur maximale de la séquence.
    dm (int): la profondeur du modèle (dimension d'embedding).

    Returns:
    numpy.ndarray: tableau de forme (max_seq_len, dm) contenant les vecteurs
                   d'encodage positionnel.
    """
    # vecteur des positions (0 à max_seq_len - 1) de forme (max_seq_len, 1)
    pos = np.arange(max_seq_len)[:, np.newaxis]

    # Création des indices des dimensions i de forme (1, dm)
    i = np.arange(dm)[np.newaxis, :]

    # Calcul des angles: pos / (10000 ^ (2 * (i // 2) / dm))
    angle_rates = pos / np.power(10000, (2 * (i // 2)) / dm)

    # Initialisation du tableau d'encodage
    PE = np.zeros((max_seq_len, dm))

    # Application du sinus aux indices pairs (2i)
    PE[:, 0::2] = np.sin(angle_rates[:, 0::2])

    # Application du cosinus aux indices impairs (2i + 1)
    PE[:, 1::2] = np.cos(angle_rates[:, 1::2])

    return PE
