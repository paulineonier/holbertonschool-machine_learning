#!/usr/bin/env python3
"""
This module defines a Poisson class that represents a Poisson distribution.
"""


class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Class constructor

        Args:
            data (list): data to estimate the distribution
            lambtha (float): expected number of occurrences
        """
        if data is None:
            # No data provided, use lambtha
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Data provided, validate
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Estimate lambtha from data
            self.lambtha = float(sum(data) / len(data))
