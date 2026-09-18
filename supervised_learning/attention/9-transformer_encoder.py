#!/usr/bin/env python3
"""Module pour la création de l'encodeur d'un Transformer."""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Représente l'encodeur complet pour une architecture Transformer."""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initialise l'encodeur.

        Parameters:
        N (int): nombre de blocs dans l'encodeur.
        dm (int): dimensionnalité du modèle.
        h (int): nombre de têtes d'attention.
        hidden (int): nombre d'unités dans les couches denses cachées.
        input_vocab (int): taille du vocabulaire d'entrée.
        max_seq_len (int): longueur maximale de séquence possible.
        drop_rate (float): taux de dropout.
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Passe avant (forward pass) de l'encodeur.

        Parameters:
        x (tf.Tensor): tensor de forme (batch, input_seq_len) contenant
                       les indices des mots.
        training (bool): indique si le modèle est en cours d'entraînement.
        mask (tf.Tensor): masque à appliquer pour le Multi-Head Attention.

        Returns:
        tf.Tensor: tensor de forme (batch, input_seq_len, dm) contenant
                   la sortie de l'encodeur.
        """
        seq_len = tf.shape(x)[1]

        # 1. Embedding + mise à l'échelle par sqrt(dm)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # 2. Ajout des encodages positionnels
        x += self.positional_encoding[:seq_len, :]

        # 3. Dropout initial
        x = self.dropout(x, training=training)

        # 4. Passage dans les N blocs
        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x
