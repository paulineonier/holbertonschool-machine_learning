#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Plots a scatter plot of men's height vs weight with magenta points.
    """
    np.random.seed(5)
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180

    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(x, y, color='magenta')
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")
    plt.title("Men’s Height vs Weight")
    plt.show()


scatter()
