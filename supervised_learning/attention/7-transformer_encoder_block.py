#!/usr/bin/env python3
"""Module pour la création d'un bloc encodeur de Transformer."""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Représente un bloc encodeur pour une architecture Transformer."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialise le bloc encodeur.

        Parameters:
        dm (int): dimensionnalité du modèle.
        h (int): nombre de têtes d'attention.
        hidden (int): nombre d'unités dans la couche dense cachée.
        drop_rate (float): taux de dropout.
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Passe avant (forward pass) du bloc encodeur.

        Parameters:
        x (tf.Tensor): tensor de forme (batch, input_seq_len, dm)
                       contenant l'entrée du bloc.
        training (bool): indique si le modèle est en cours d'entraînement.
        mask (tf.Tensor, optional): masque pour le MultiHeadAttention.

        Returns:
        tf.Tensor: tensor de forme (batch, input_seq_len, dm)
                   contenant la sortie du bloc encodeur.
        """
        # 1. Multi-Head Attention + Dropout
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # Première connexion résiduelle + Layer Normalization
        out1 = self.layernorm1(x + attn_output)

        # 2. Position-wise Feed Forward Network (FFN) + Dropout
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Seconde connexion résiduelle + Layer Normalization
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
