#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def line():
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 11)
    plt.plot(x, y, 'r-')  # Solid red line
    plt.xlim(0, 10)       # x-axis goes from 0 to 10
    plt.show()
