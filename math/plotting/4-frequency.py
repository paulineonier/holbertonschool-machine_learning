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

    # Plot histogram with bins every 10 units and black edges
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

    # Configure the graph
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.xticks(range(0, 101, 10))
    plt.xlim(0, 100)

    plt.show()
