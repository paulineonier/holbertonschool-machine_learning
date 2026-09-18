#!/usr/bin/env python3
"""Module pour la création d'un bloc décodeur de Transformer."""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Représente un bloc décodeur pour une architecture Transformer."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialise le bloc décodeur.

        Parameters:
        dm (int): dimensionnalité du modèle.
        h (int): nombre de têtes d'attention.
        hidden (int): nombre d'unités dans la couche dense cachée.
        drop_rate (float): taux de dropout.
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Passe avant (forward pass) du bloc décodeur.

        Parameters:
        x (tf.Tensor): tensor de forme (batch, target_seq_len, dm)
                       contenant l'entrée du décodeur.
        encoder_output (tf.Tensor): tensor de forme (batch, input_seq_len, dm)
                                   provenant de l'encodeur.
        training (bool): indique si le modèle est en cours d'entraînement.
        look_ahead_mask (tf.Tensor): masque causale pour la première MHA.
        padding_mask (tf.Tensor): masque de rembourrage pour la seconde MHA.

        Returns:
        tf.Tensor: tensor de forme (batch, target_seq_len, dm)
                   contenant la sortie du bloc décodeur.
        """
        # 1. Masked Multi-Head Self-Attention (Q=K=V=x) + Dropout + ResNet & LN
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # 2. Encoder-Decoder Attention
        attn2, _ = self.mha2(out1, encoder_output,
                             encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # 3. Feed Forward Network (FFN) + Dropout + ResNet & LN
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3
