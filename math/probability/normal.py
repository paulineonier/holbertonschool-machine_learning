#!/usr/bin/env python3
"""Module définit une classe Normal représente une distrib normale"""


class Normal:
    """Classe représentant une distribution normale"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialise la distribution normale

        Args:
            data (list): liste de données pour estimer la distribution
            mean (float): moyenne de la distribution
            stddev (float): écart-type de la distribution

        Raises:
            TypeError: si data n'est pas une liste
            ValueError: si data contient moins de 2 valeurs
            ValueError: si stddev <= 0
        """

        # Cas où data n'est pas fourni
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")

            self.mean = float(mean)
            self.stddev = float(stddev)

        # Cas où data est fourni
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calcul de la moyenne
            self.mean = float(sum(data) / len(data))

            # Calcul de la variance
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)

            # Calcul de l'écart-type
            self.stddev = float(variance ** 0.5)
