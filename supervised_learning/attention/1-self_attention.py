#!/usr/bin/env python3
"""Module pour le calcul du mécanisme d'attention (Bahdanau Attention)."""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calcule le mécanisme d'attention pour la traduction automatique."""

    def __init__(self, units):
        """Initialise la couche d'attention.

        Parameters:
        units (int): nombre d'unités cachées dans le modèle d'alignement.
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Calcule le vecteur de contexte et les poids d'attention.

        Parameters:
        s_prev (tf.Tensor): état caché précédent du décodeur,
                            de forme (batch, units).
        hidden_states (tf.Tensor): états cachés de l'encodeur,
                                   de forme (batch, input_seq_len, units).

        Returns:
        tuple: (context, weights)
               - context: tensor de forme (batch, units)
               - weights: tensor de forme (batch, input_seq_len, 1)
        """
        # pour permettre le broadcasting avec hidden_states
        s_prev_expanded = tf.expand_dims(s_prev, axis=1)

        # Calcul du score d'alignement
        score = self.V(tf.nn.tanh(self.W(s_prev_expanded)
                                  + self.U(hidden_states)))

        # Application du Softmax le long de l'axe de la séquence (axis=1)
        weights = tf.nn.softmax(score, axis=1)

        # Calcul du vecteur de contexte par somme pondérée hidden_states
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
