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

            # Mean of data
            mean = sum(data) / len(data)

            # Variance of data
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # Step 1: estimate p
            p = 1 - (variance / mean)

            # Step 2: estimate n (rounded, not casted)
            n = round(mean / p)

            # Step 3: recompute p for consistency
            p = mean / n

            self.n = int(n)
            self.p = float(p)
