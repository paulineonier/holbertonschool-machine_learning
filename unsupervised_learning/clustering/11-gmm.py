#!/usr/bin/env python3
"""Gaussian Mixture Model"""

import sklearn.mixture


def gmm(X, k):
    """
    Calcule un GMM sur un dataset

    Paramètres:
    X : ndarray (n, d)
    k : nombre de clusters

    Retour:
    pi : (k,)
    m : (k, d)
    S : (k, d, d)
    clss : (n,)
    bic : float
    """

    try:
        # Vérifications de base
        if not hasattr(X, "shape") or len(X.shape) != 2:
            return None, None, None, None, None

        n, d = X.shape

        if not isinstance(k, int) or k <= 0 or k > n:
            return None, None, None, None, None

        # Création du modèle GMM
        gmm = sklearn.mixture.GaussianMixture(n_components=k)

        # Entraînement
        gmm.fit(X)

        # Extraction des paramètres
        pi = gmm.weights_
        m = gmm.means_
        S = gmm.covariances_

        # Prédiction des clusters
        clss = gmm.predict(X)

        # Calcul du BIC
        bic = gmm.bic(X)

        return pi, m, S, clss, bic

    except Exception:
        return None, None, None, None, None
