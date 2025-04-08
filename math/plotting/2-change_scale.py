#!/usr/bin/env python3
"""
Module plots exponential decay C-14 with logarithmic scale for y-axis.
It uses a line graph with the x-axis ranging from 0 to 28650 years.
"""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plots the exponential decay of C-14 with a logarithmic scale on the y-axis.
    X-axis represents time in years and y-axis represents fraction remaining.
    """
    x = np.linspace(0, 28650, 1000)  # Time points in years
    r = np.log(0.5)  # Decay constant (logarithm of 0.5)
    t = 5730  # Half-life of C-14 in years
    y = np.exp((r / t) * x)  # Exponential decay formula

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y)  # Plotting the data as a blue solid line
    plt.xlabel('Time (years)')  # Label for the x-axis
    plt.ylabel('Fraction Remaining')  # Label for the y-axis
    plt.title('Exponential Decay of C-14')  # Title of the graph
    plt.yscale('log')  # Set the y-axis to a logarithmic scale
    plt.xlim(0, 28650)  # Ensure the x-axis goes from 0 to 28650
    plt.ylim(0.0001, 1)
    plt.show()  # Display the plot
