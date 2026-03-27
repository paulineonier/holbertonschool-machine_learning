#!/usr/bin/env python3
"""Agglomerative clustering"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt
import numpy as np


def agglomerative(X, dist):
    """
    Effectue un clustering hiérarchique agglomératif

    Paramètres:
    X : ndarray (n, d)
    dist : distance max (cophenetic distance)

    Retour:
    clss : ndarray (n,)
    """

    try:
        # Vérifications de base
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None

        if not isinstance(dist, (int, float)) or dist < 0:
            return None

        # Calcul de la matrice de liaison (linkage matrix)
        Z = scipy.cluster.hierarchy.linkage(X, method='ward')

        # Attribution des clusters selon la distance seuil
        clss = scipy.cluster.hierarchy.fcluster(Z, t=dist,
                                                criterion='distance')

        # Affichage du dendrogramme
        plt.figure()

        scipy.cluster.hierarchy.dendrogram(
            Z,
            color_threshold=dist  # couleurs différentes selon clusters
        )

        plt.axhline(y=dist, color='red', linestyle='--')
        plt.title("Dendrogramme (clustering agglomératif)")
        plt.xlabel("Data points")
        plt.ylabel("Distance")

        plt.show()

        return clss

    except Exception:
        return None
