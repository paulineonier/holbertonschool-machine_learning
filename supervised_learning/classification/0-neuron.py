# Fichier : 0-neuron.py

import numpy as np

class Neuron:
    """Classe qui définit un neurone pour la classification binaire."""

    def __init__(self, nx):
        """
        nx : nombre de caractéristiques en entrée.
        """
        # Étape 1 : Vérifications
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Étape 2 : Attributs publics
        # Poids W : array de shape (1, nx), aléatoire selon une loi normale
        self.W = np.random.randn(1, nx)

        # Biais b : initialisé à 0
        self.b = 0

        # Sortie activée A : initialisée à 0
        self.A = 0
