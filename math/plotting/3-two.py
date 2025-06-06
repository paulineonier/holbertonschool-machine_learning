#!/usr/bin/env python3
"""
This module contains the function to plot the exponential decay of two
radioactive elements: C-14 and Ra-226. The plot displays how the fraction
remaining of each element decreases over time.
"""

import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    Plots exponential decay for two radioactive elements (C-14 and Ra-226).
    """
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y1, 'r--', label="C-14")  # Dashed red line for C-14
    plt.plot(x, y2, 'g-', label="Ra-226")  # Solid green line for Ra-226
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of Radioactive Elements")
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.show()
