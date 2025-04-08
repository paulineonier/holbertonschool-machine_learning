#!/usr/bin/env python3
"""
This module contains a function that plots all 5 previous graphs in one figure.
The plots are arranged in a 3x2 grid, with the last plot occupying two columns.
"""

import numpy as np
import matplotlib.pyplot as plt

def all_in_one():
    """
    This function plots all 5 previous graphs in one figure.
    The figure has a 3x2 grid layout, and the last plot occupies two columns.
    All axis labels and plot titles are set to x-small font size.
    """
    
    # 1st Plot: y = x³
    x0 = np.arange(0, 11)
    y0 = x0 ** 3
    
    # 2nd Plot: Scatter plot (Height vs Weight)
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180
    
    # 3rd Plot: Exponential Decay of C-14
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)
    
    # 4th Plot: Exponential Decay of Radioactive Elements
    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)
    
    # 5th Plot: Histogram of Student Grades
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # Create a 3x2 grid of subplots
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle("All in One", fontsize="x-small")  # Title of the entire figure

    # Plot 1: y = x³ (in the first subplot)
    axes[0, 0].plot(x0, y0, 'r-', label="y = x³")
    axes[0, 0].set_xlabel("x", fontsize="x-small")
    axes[0, 0].set_ylabel("y", fontsize="x-small")
    axes[0, 0].set_title("Graph of y = x³", fontsize="x-small")
    axes[0, 0].tick_params(axis='both', labelsize='x-small')

    # Plot 2: Scatter plot (Height vs Weight) (in the second subplot)
    axes[0, 1].scatter(x1, y1, color='magenta', label="Height vs Weight")
    axes[0, 1].set_xlabel("Height (in)", fontsize="x-small")
    axes[0, 1].set_ylabel("Weight (lbs)", fontsize="x-small")
    axes[0, 1].set_title("Men's Height vs Weight", fontsize="x-small")
    axes[0, 1].tick_params(axis='both', labelsize='x-small')

    # Plot 3: Exponential Decay of C-14 (in the third subplot)
    axes[1, 0].plot(x2, y2, 'b-', label="Exponential Decay of C-14")
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel("Time (years)", fontsize="x-small")
    axes[1, 0].set_ylabel("Fraction Remaining", fontsize="x-small")
    axes[1, 0].set_title("Exponential Decay of C-14", fontsize="x-small")
    axes[1, 0].tick_params(axis='both', labelsize='x-small')

    # Plot 4: Exponential Decay of Radioactive Elements (in the fourth subplot)
    axes[1, 1].plot(x3, y31, 'r--', label="C-14")
    axes[1, 1].plot(x3, y32, 'g-', label="Ra-226")
    axes[1, 1].set_xlabel("Time (years)", fontsize="x-small")
    axes[1, 1].set_ylabel("Fraction Remaining", fontsize="x-small")
    axes[1, 1].set_title("Exponential Decay of Radioactive Elements", fontsize="x-small")
    axes[1, 1].set_xlim(0, 20000)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(loc="upper right", fontsize="x-small")
    axes[1, 1].tick_params(axis='both', labelsize='x-small')

    # Plot 5: Histogram of Student Grades (in the last plot, taking up two columns)
    axes[2, 0].hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    axes[2, 0].set_xlabel("Grades", fontsize="x-small")
    axes[2, 0].set_ylabel("Number of Students", fontsize="x-small")
    axes[2, 0].set_title("Project A", fontsize="x-small")
    axes[2, 0].tick_params(axis='both', labelsize='x-small')

    # Remove the empty subplot on the last row, second column
    fig.delaxes(axes[2, 1])

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Adjust space for the main title
    plt.show()  # Display the combined figure
