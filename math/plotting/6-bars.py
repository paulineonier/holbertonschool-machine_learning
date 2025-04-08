#!/usr/bin/env python3
"""
This module generates a stacked bar graph showing the number of fruits
each person possesses. Each fruit type is represented by a specific color,
and the bars are stacked by fruit type.
"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plots a stacked bar graph representing the quantity of fruits
    owned by Farrah, Fred, and Felicia.
    """
    # Generate random data for fruit quantities
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))  # Fruit matrix

    # Data for each person
    people = ['Farrah', 'Fred', 'Felicia']
    apples = fruit[0]  # First row: apples
    bananas = fruit[1]  # Second row: bananas
    oranges = fruit[2]  # Third row: oranges
    peaches = fruit[3]  # Fourth row: peaches

    # Define bar width
    bar_width = 0.5

    # Plot stacked bars
    plt.bar(people, apples, bar_width, color='red', label='apples')
    plt.bar(
        people, bananas, bar_width, bottom=apples, color='yellow',
        label='bananas'
    )
    plt.bar(
        people, oranges, bar_width, bottom=apples + bananas, color='#ff8000',
        label='oranges'
    )
    plt.bar(
        people, peaches, bar_width, bottom=apples + bananas + oranges,
        color='#ffe5b4', label='peaches'
    )

    # Configure the graph
    plt.ylabel('Quantity of Fruit')  # Label for the y-axis
    plt.title('Number of Fruit per Person')  # Title of the graph
    plt.ylim(0, 80)  # Set y-axis range from 0 to 80
    plt.yticks(range(0, 81, 10))  # Set ticks every 10 units
    plt.xticks(range(len(people)), people)  # Ensure x-axis labels match people
    plt.legend()  # Display the legend
    plt.show()  # Display the graph
