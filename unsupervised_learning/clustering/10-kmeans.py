#!/usr/bin/env python3
"""K-means clustering"""

import sklearn.cluster


def kmeans(X, k):
    """
    Effectue le clustering K-means

    Paramètres:
    X : ndarray (n, d) -> dataset
    k : nombre de clusters

    Retour:
    C : centroïdes (k, d)
    clss : indices des clusters (n,)
    """

    try:
        # Vérifications de base
        if not isinstance(X, (list, tuple)) and not hasattr(X, "shape"):
            return None, None

        if len(X.shape) != 2:
            return None, None

        n, d = X.shape

        if not isinstance(k, int) or k <= 0 or k > n:
            return None, None

        # Création du modèle KMeans
        kmeans = sklearn.cluster.KMeans(n_clusters=k)

        # Apprentissage du modèle
        kmeans.fit(X)

        # Récupération des résultats
        C = kmeans.cluster_centers_
        clss = kmeans.labels_

        return C, clss

    except Exception:
        return None, None
