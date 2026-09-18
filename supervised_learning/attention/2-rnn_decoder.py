#!/usr/bin/env python3
"""Module de décodeur RNN pour la traduction automatique."""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Décodeur basé sur un GRU avec mécanisme d'attention."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialise les attributs du décodeur.

        Parameters:
        vocab (int): taille du vocabulaire de sortie.
        embedding (int): dimension des vecteurs d'embedding.
        units (int): nombre d'unités cachées de la cellule GRU.
        batch (int): taille du batch.
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Passe avant (forward pass) du décodeur pour un pas de temps.

        Parameters:
        x (tf.Tensor): tensor de forme (batch, 1) contenant l'indice du mot.
        s_prev (tf.Tensor): état caché précédent du décodeur (batch, units).
        hidden_states (tf.Tensor): états cachés de l'encodeur
                                   (batch, input_seq_len, units).

        Returns:
        tuple: (y, s) où:
               - y: tensor de forme (batch, vocab) des probabilités du mot
               - s: nouvel état caché du décodeur de forme (batch, units)
        """
        # 1. Calcul du vecteur de contexte via la couche d'attention
        context, _ = self.attention(s_prev, hidden_states)

        # 2. Embedding du mot d'entrée (batch, 1) -> (batch, 1, embedding)
        x = self.embedding(x)

        # Concatenation du vecteur de contexte et de l'embedding dans cet ordre
        # context= forme (batch, units), + ajoute dimension pr correspondre à x
        context_expanded = tf.expand_dims(context, axis=1)
        x = tf.concat([context_expanded, x], axis=-1)

        # 4. Passage dans le GRU
        output, s = self.gru(x, initial_state=s_prev)

        # 5. Redimensionnement de la sortie pour la couche dense
        # output est de forme (batch, 1, units) -> (batch, units)
        output = tf.reshape(output, (-1, output.shape[2]))

        # 6. Projections dense vers la taille du vocabulaire
        y = self.F(output)

        return y, s
