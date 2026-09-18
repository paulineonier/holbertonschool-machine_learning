#!/usr/bin/env python3
"""Module d'encodeur RNN pour la traduction automatique."""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Encodeur basé sur un GRU pour une architecture Encoder-Decoder."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialise les attributs de l'encodeur.

        Parameters:
        vocab (int): taille du vocabulaire d'entrée.
        embedding (int): dimension des vecteurs d'embedding.
        units (int): nombre d'unités cachées de la cellule GRU.
        batch (int): taille du batch.
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """Initialise l'état caché avec des zéros.

        Returns:
        tf.Tensor: tensor de forme (batch, units) rempli de zéros.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Passe avant (forward pass) de l'encodeur.

        Parameters:
        x (tf.Tensor): tensor de forme (batch, input_seq_len) contenant les
                       indices des mots.
        initial (tf.Tensor): état caché initial de forme (batch, units).

        Returns:
        tuple: (outputs, hidden) où:
               - outputs: tensor de forme (batch, input_seq_len, units)
               - hidden: tensor de forme (batch, units) de l'état caché final
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
