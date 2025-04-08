#!/usr/bin/env python3
"""
This module plots a histogram of student scores for Project A.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student scores for Project A.

    The histogram displays the distribution of student grades, with:
    - The x-axis labeled "Grades"
    - The y-axis labeled "Number of Students"
    - Bins set every 10 units, ranging from 0 to 100
    - The bars outlined in black to make them clearly visible

    Random distribution student grades generated using normal distribution
    with a mean of 68 and a standard deviation of 15.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))  # Set the figure size

    # Plotting the histogram with bins every 10 units
    plt.hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')

    # Adding labels and title
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
