#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plots the exponential decay of C-14 with a logarithmically scaled y-axis.
    """
    x = np.arange(0, 28651, 5730)  # Time points in years
    r = np.log(0.5)               # Decay constant (logarithm of 0.5)
    t = 5730                      # Half-life of C-14 in years
    y = np.exp((r / t) * x)       # Exponential decay formula

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, 'b-')          # 'b-' specifies a blue solid line
    plt.xlabel('Time (years)')    # Label for the x-axis
    plt.ylabel('Fraction Remaining')  # Label for the y-axis
    plt.title('Exponential Decay of C-14')  # Title of the graph
    plt.yscale('log')             # Set the y-axis to a logarithmic scale
    plt.grid(True)                # Add grid lines for better readability
    plt.show()                    # Display the graph
