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
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def factorial(self, n):
        """Calculates factorial without using math module"""
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def exp(self, x):
        """Calculates e^x using Taylor series with high precision"""
        term = 1.0
        sum_ = 1.0
        for i in range(1, 100):  # 100 terms for high precision
            term *= x / i
            sum_ += term
        return sum_

    def pmf(self, k):
        """Calculates the PMF for a given number of successes k"""
        k = int(k)
        if k < 0:
            return 0
        return (self.exp(-self.lambtha) * self.lambtha**k) / self.factorial(k)

    def cdf(self, k):
        """Calculates the CDF for a given number of successes k"""
        k = int(k)
        if k < 0:
            return 0
        total = 0.0
        for i in range(k + 1):
            total += self.pmf(i)
        return total
