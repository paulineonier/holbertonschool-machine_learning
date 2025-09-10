#!/usr/bin/env python3
"""
Module 2-l2_reg_cost
This module defines a function to calculate the cost of a Keras model
with L2 regularization.

Functions:
    l2_reg_cost(cost, model):
        Calculates the total cost per layer of a Keras model with L2
        regularization.
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost (tf.Tensor): Tensor containing the cost of the network
                          without L2 regularization.
        model (tf.keras.Model): A Keras model that includes layers with
                                L2 regularization.

    Returns:
        tf.Tensor: A 1D tensor containing the total cost for each
                   layer of the network, accounting for L2
                   regularization.
    """
    reg_losses = tf.convert_to_tensor(model.losses)
    total_costs = cost + reg_losses
    return total_costs
