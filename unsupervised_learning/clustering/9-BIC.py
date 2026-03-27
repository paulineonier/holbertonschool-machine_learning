#!/usr/bin/env python3
"""Fonction BIC pour choisir le meilleur nombre de clusters"""

import numpy as np

# Import de la fonction EM fournie
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000,
        tol=1e-5, verbose=False):
    """
    Détermine le meilleur nombre de clusters pour un GMM
    en utilisant le critère BIC

    Paramètres:
    X : ndarray (n, d) -> dataset
    kmin : nombre minimum de clusters
    kmax : nombre maximum de clusters
    iterations : nombre max d'itérations EM
    tol : tolérance pour convergence
    verbose : affichage EM

    Retour:
    best_k : meilleur nombre de clusters
    best_result : (pi, m, S)
    l : log likelihoods pour chaque k
    b : BIC pour chaque k
    """

    try:
        # Vérifications de base
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None, None, None, None

        n, d = X.shape

        if not isinstance(kmin, int) or kmin <= 0:
            return None, None, None, None

        if kmax is None:
            kmax = n

        if not isinstance(kmax, int) or kmax < kmin:
            return None, None, None, None

        # Tableaux pour stocker résultats
        ks = kmax - kmin + 1
        l = np.zeros(ks)  # log likelihoods
        b = np.zeros(ks)  # BIC

        best_bic = None
        best_k = None
        best_result = None

        # 🔁 SEULE boucle autorisée
        for i, k in enumerate(range(kmin, kmax + 1)):

            # EM pour ce nombre de clusters
            pi, m, S, log_likelihood = expectation_maximization(
                X, k, iterations=iterations, tol=tol, verbose=verbose
            )

            # Sauvegarde du log likelihood
            l[i] = log_likelihood

            # Calcul du nombre de paramètres p
            p = (k * d) + (k * d * (d + 1) / 2) + (k - 1)

            # Calcul du BIC
            bic = p * np.log(n) - 2 * log_likelihood
            b[i] = bic

            # Mise à jour du meilleur modèle
            if best_bic is None or bic < best_bic:
                best_bic = bic
                best_k = k
                best_result = (pi, m, S)

        return best_k, best_result, l, b

    except Exception:
        return None, None, None, None
