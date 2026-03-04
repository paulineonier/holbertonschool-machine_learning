#!/usr/bin/env python3
"""Module that defines a Normal distribution class"""


class Normal:
    """Class representing a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the normal distribution

        Args:
            data (list): list of data to estimate the distribution
            mean (float): mean of the distribution
            stddev (float): standard deviation of the distribution

        Raises:
            TypeError: if data is not a list
            ValueError: if data contains less than 2 values
            ValueError: if stddev is not a positive value
        """

        # Case where data is not provided
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")

            self.mean = float(mean)
            self.stddev = float(stddev)

        # Case where data is provided
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean
            self.mean = float(sum(data) / len(data))

            # Calculate variance
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)

            # Calculate standard deviation
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value

        Args:
            x (float): the x-value

        Returns:
            float: the z-score
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score

        Args:
            z (float): the z-score

        Returns:
            float: the x-value
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Calculates the PDF value for a given x

        Args:
            x (float): the x-value

        Returns:
            float: the PDF value
        """
        e = 2.7182818285
        pi = 3.1415926536

        return (1 / (self.stddev * (2 * pi) ** 0.5)) * \
               (e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2))

    def cdf(self, x):
        """
        Calculates the CDF value for a given x

        Args:
            x (float): the x-value

        Returns:
            float: the CDF value
        """
        pi = 3.1415926536

        # Calculate z-score
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # Approximation of erf (Abramowitz and Stegun formula)
        t = 1 / (1 + 0.3275911 * abs(z))

        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429

        erf = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * \
            (2.7182818285 ** (-z * z))

        # Adjust sign
        if z < 0:
            erf = -erf

        return 0.5 * (1 + erf)
