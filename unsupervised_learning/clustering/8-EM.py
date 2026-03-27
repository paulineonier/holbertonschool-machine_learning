#!/usr/bin/env python3
"""Expectation Maximization pour un GMM"""

import numpy as np

# Imports imposés
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000,
                             tol=1e-5, verbose=False):
    """
    Effectue l'algorithme EM pour un GMM

    Paramètres:
    X : ndarray (n, d) -> dataset
    k : nombre de clusters
    iterations : nombre max d'itérations
    tol : tolérance pour arrêt
    verbose : affichage des logs

    Retour:
    pi : proportions (k,)
    m : moyennes (k, d)
    S : covariances (k, d, d)
    g : responsabilités (k, n)
    L : log likelihood final
    """

    try:
        # Vérifications de base
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None, None, None, None, None

        n, d = X.shape

        if not isinstance(k, int) or k <= 0 or k > n:
            return None, None, None, None, None

        # Initialisation des paramètres
        pi, m, S = initialize(X, k)

        # Première expectation pour initialiser l
        g, L = expectation(X, pi, m, S)

        # 🔁 SEULE boucle autorisée
        for i in range(iterations):
            # M-step : mise à jour des paramètres
            pi, m, S = maximization(X, g)

            # E-step : recalcul des probabilités
            g, new_l = expectation(X, pi, m, S)

            # Affichage si verbose
            if verbose and (i % 10 == 0):
                print("Log Likelihood after {} iterations: {:.5f}".format(i, new_l))

            # Condition d'arrêt
            if abs(new_l - l) <= tol:
                if verbose:
                    print("Log Likelihood after {} iterations: {:.5f}".format(i, new_l))
                return pi, m, S, g, new_l

            # Mise à jour du log likelihood
            L = new_l

        # Affichage final si non déjà affiché
        if verbose:
            print("Log Likelihood after {} iterations: {:.5f}".format(iterations, L))

        return pi, m, S, g, L

    except Exception:
        return None, None, None, None, None
