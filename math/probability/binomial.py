#!/usr/bin/env python3
"""Binomial distribution module"""


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Initializes the binomial distribution"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not 0 < p < 1:
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            p = 1 - (variance / mean)
            n = round(mean / p)

            self.n = int(n)
            self.p = float(p)

    def factorial(self, x):
        """Calculates factorial of x"""
        if x == 0 or x == 1:
            return 1
        return x * self.factorial(x - 1)

    def pmf(self, k):
        """Calculates PMF for k"""
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        comb = self.factorial(self.n) / (
            self.factorial(k) * self.factorial(self.n - k)
        )

        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculates CDF for k"""
        k = int(k)

        if k < 0:
            return 0
        if k >= self.n:
            return 1

        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)

        return cdf
