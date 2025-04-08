#!/usr/bin/env python3
"""
This module generates a histogram of student scores for a project.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student grades with bars outlined in black.
    The x-axis represents grades, the y-axis represents the number
    of students, and the title is 'Project A'.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Plot histogram with black outlines and blue color
    plt.hist(
        student_grades, bins=range(0, 101, 10), edgecolor='black',
        color='#1E90FF'
    )

    # Configure the graph
    plt.xlabel('Grades')  # Label for the x-axis
    plt.ylabel('Number of Students')  # Label for the y-axis
    plt.title('Project A')  # Title of the graph
    plt.xticks(range(0, 101, 10))  # Ensure ticks every 10 on x-axis
    plt.yticks(range(0, 31, 5))  # Set y-axis ticks from 0 to 30 in increments of 5
    plt.ylim(0, 30)  # Ensure y-axis range is exactly 0 to 30
    plt.show()  # Display the histogram
