#!/usr/bin/env python3
"""Module pour l'implémentation du mécanisme de Multi-Head Attention."""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Couche de Multi-Head Attention pour un Transformer."""

    def __init__(self, dm, h):
        """Initialise la couche Multi-Head Attention.

        Parameters:
        dm (int): dimensionnalité du modèle.
        h (int): nombre de têtes d'attention.
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Divise la dernière dimension en (h, depth) et transpose.

        Résultat de forme: (batch, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Passe avant du mécanisme Multi-Head Attention.

        Parameters:
        Q (tf.Tensor): tensor de forme (batch, seq_len_q, dk)
        K (tf.Tensor): tensor de forme (batch, seq_len_v, dk)
        V (tf.Tensor): tensor de forme (batch, seq_len_v, dv)
        mask (tf.Tensor): masque (toujours None selon la consigne)

        Returns:
        tuple: (output, weights)
               - output: (batch, seq_len_q, dm)
               - weights: (batch, h, seq_len_q, seq_len_v)
        """
        batch_size = tf.shape(Q)[0]

        # 1. Projections linéaires initiales
        q = self.Wq(Q)  # (batch, seq_len_q, dm)
        k = self.Wk(K)  # (batch, seq_len_v, dm)
        v = self.Wv(V)  # (batch, seq_len_v, dm)

        # 2. Séparation en h têtes d'attention
        q = self.split_heads(q, batch_size)  # (batch, h, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch, h, seq_len_v, depth)
        v = self.split_heads(v, batch_size)  # (batch, h, seq_len_v, depth)

        # 3. Scaled Dot-Product Attention sur chaque tête
        scaled_attention, weights = sdp_attention(q, k, v, mask)
        # scaled_attention: (batch, h, seq_len_q, depth)
        # weights: (batch, h, seq_len_q, seq_len_v)

        # 4. Concatenation des têtes
        # Transposition vers (batch, seq_len_q, h, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # Reshape vers (batch, seq_len_q, dm)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))

        # 5. Projection linéaire finale
        output = self.linear(concat_attention)

        return output, weights
