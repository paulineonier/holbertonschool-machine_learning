#!/usr/bin/env python3
"""Module pour la création du décodeur d'un Transformer."""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Représente le décodeur complet pour une architecture Transformer."""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initialise le décodeur.

        Parameters:
        N (int): nombre de blocs dans le décodeur.
        dm (int): dimensionnalité du modèle.
        h (int): nombre de têtes d'attention.
        hidden (int): nombre d'unités dans les couches denses cachées.
        target_vocab (int): taille du vocabulaire cible.
        max_seq_len (int): longueur maximale de séquence possible.
        drop_rate (float): taux de dropout.
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Passe avant (forward pass) du décodeur.

        Parameters:
        x (tf.Tensor): tensor de forme (batch, target_seq_len) contenant
                       les indices des mots cibles.
        encoder_output (tf.Tensor): tensor de forme (batch, input_seq_len, dm)
                                   contenant la sortie de l'encodeur.
        training (bool): indique si le modèle est en cours d'entraînement.
        look_ahead_mask (tf.Tensor): masque causale pour la 1re MHA.
        padding_mask (tf.Tensor): masque de remplissage pour la 2e MHA.

        Returns:
        tf.Tensor: tensor de forme (batch, target_seq_len, dm) contenant
                   la sortie du décodeur.
        """
        seq_len = tf.shape(x)[1]

        # 1. Embedding + mise à l'échelle par sqrt(dm)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # 2. Ajout des encodages positionnels
        x += self.positional_encoding[:seq_len, :]

        # 3. Application du dropout initial
        x = self.dropout(x, training=training)

        # 4. Passage successif à travers les N blocs de décodeur
        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training,
                               look_ahead_mask, padding_mask)

        return x
