#!/usr/bin/env python3
"""
This module defines the function scatter, which plots a scatter plot
of men's height vs weight with magenta points.
"""

import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Plots a scatter plot of men's height vs weight with magenta points.
    """
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180

    plt.figure(figsize=(6.4, 4.8))  # Ensure the figure size is correct
    plt.scatter(x, y, c='magenta')
    plt.xlabel('Height (in)')  # Label for x-axis
    plt.ylabel('Weight (lbs)')  # Label for y-axis
    plt.title("Men's Height vs Weight")  # Title of the plot
    plt.show()  # Display the plot


scatter()
