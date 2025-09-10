#!/usr/bin/env python3
"""
7-early_stopping.py
Determines whether to stop training early based on validation cost.
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Checks if gradient descent should stop early.

    Args:
        cost (float): current validation cost of the network
        opt_cost (float): lowest recorded validation cost
        threshold (float): threshold for early stopping
        patience (int): patience count for early stopping
        count (int): current count of how long threshold has not been met

    Returns:
        Tuple (stop, new_count):
            stop (bool): True if training should stop
            new_count (int): updated patience count
    """
    # If the improvement is less than the threshold, increment count
    if opt_cost - cost > threshold:
        # Cost decreased significantly, reset counter
        count = 0
    else:
        # Cost did not improve enough, increment counter
        count += 1

    # Determine if early stopping should occur
    stop = count >= patience

    return stop, count
