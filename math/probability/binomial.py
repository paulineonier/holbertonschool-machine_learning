#!/usr/bin/env python3
"""Module that defines a Binomial distribution class"""


class Binomial:
    """Class representing a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the binomial distribution

        Args:
            data (list): list of data to estimate the distribution
            n (int): number of Bernoulli trials
            p (float): probability of success

        Raises:
            TypeError: if data is not a list
            ValueError: if data contains less than 2 values
            ValueError: if n is not positive
            ValueError: if p is not valid
        """

        # Case where data is not provided
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")

            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = int(n)
            self.p = float(p)

        # Case where data is provided
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean
            mean = sum(data) / len(data)

            # Calculate variance
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # Estimate p
            p = 1 - (variance / mean)

            # Estimate n (rounded)
            n = round(mean / p)

            # Recalculate p
            p = mean / n

            self.n = int(n)
            self.p = float(p)

    def pmf(self, k):
        """
        Calculates the PMF value for a given number of successes

        Args:
            k (int): number of successes

        Returns:
            float: the PMF value
        """

        # Convert k to integer
        k = int(k)

        # Check range
        if k < 0 or k > self.n:
            return 0

        # Factorial function
        def factorial(n):
            if n == 0 or n == 1:
                return 1
            return n * factorial(n - 1)

        # Compute combination
        nCk = factorial(self.n) / (factorial(k) * factorial(self.n - k))

        # Apply binomial formula
        return nCk * (self.p ** k) * ((1 - self.p) ** (self.n - k))
