#!/usr/bin/env python3
"""Module pour l'assemblage complet du modèle Transformer."""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """Réseau complet Transformer héritant de tf.keras.Model."""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """Initialise le modèle Transformer.

        Parameters:
        N (int): nombre de blocs dans l'encodeur et le décodeur.
        dm (int): dimensionnalité du modèle.
        h (int): nombre de têtes d'attention.
        hidden (int): nombre d'unités dans les couches denses cachées.
        input_vocab (int): taille du vocabulaire d'entrée.
        target_vocab (int): taille du vocabulaire cible.
        max_seq_input (int): longueur maximale de séquence pour l'entrée.
        max_seq_target (int): longueur maximale de séquence pour la cible.
        drop_rate (float): taux de dropout.
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """Passe avant (forward pass) du Transformer.

        Parameters:
        inputs (tf.Tensor): tensor de forme (batch, input_seq_len)
        target (tf.Tensor): tensor de forme (batch, target_seq_len)
        training (bool): indique si le modèle est en cours d'entraînement
        encoder_mask (tf.Tensor): masque d'entrée pour l'encodeur
        look_ahead_mask (tf.Tensor): masque causal pour le décodeur
        decoder_mask (tf.Tensor): masque de remplissage pour la 2e MHA
                                 du décodeur (cross-attention)

        Returns:
        tf.Tensor: tensor de forme (batch, target_seq_len, target_vocab)
                   contenant les logits de sortie.
        """
        # 1. Traitement dans l'encodeur
        enc_output = self.encoder(inputs, training, encoder_mask)

        # 2. Traitement dans le décodeur
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)

        # 3. Projection linéaire vers les logits du vocabulaire cible
        final_output = self.linear(dec_output)

        return final_output
