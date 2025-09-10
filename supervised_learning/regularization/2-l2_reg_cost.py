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
        tf.Tensor: A tensor containing the total cost for each layer
                   of the network, accounting for L2 regularization.
    """
    # Liste des coûts de régularisation (un par couche)
    reg_losses = model.losses

    # On crée une liste où chaque élément = cost + perte L2 (0.0 si pas de régul)
    per_layer_costs = [
        cost + loss for loss in reg_losses
    ]

    # S’assurer que toutes les couches sont comptées
    # (y compris celles sans régularisation -> ajout 0.0)
    if len(per_layer_costs) < len(model.layers):
        per_layer_costs.extend([cost + 0.0] * (len(model.layers) - len(per_layer_costs)))

    # Retourne un tenseur contenant tous les coûts
    return tf.convert_to_tensor(per_layer_costs)
