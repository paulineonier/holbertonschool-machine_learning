#!/usr/bin/env python3
"""
Module defines an Exponential class that represents exponential distribution.
"""


class Exponential:
    """Represents an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Class constructor

        Args:
            data (list): data to estimate the distribution
            lambtha (float): expected number of occurrences
        """
        if data is None:
            # Use provided lambtha
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Validate data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Estimate lambtha from data
            mean = sum(data) / len(data)
            if mean == 0:
                raise ValueError("mean of data must be positive")
            self.lambtha = 1 / mean
