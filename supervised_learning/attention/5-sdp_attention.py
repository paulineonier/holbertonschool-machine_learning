#!/usr/bin/env python3
"""Module pour le calcul de la Scaled Dot-Product Attention."""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calcule la Scaled Dot-Product Attention.

    Parameters:
    Q (tf.Tensor): matrice des requêtes (..., seq_len_q, dk).
    K (tf.Tensor): matrice des clés (..., seq_len_v, dk).
    V (tf.Tensor): matrice des valeurs (..., seq_len_v, dv).
    mask (tf.Tensor, optional): masque optionnel diffusible en
                                (..., seq_len_q, seq_len_v). Défaut à None.

    Returns:
    tuple: (output, weights)
           - output: tensor de forme (..., seq_len_q, dv)
           - weights: tensor de forme (..., seq_len_q, seq_len_v)
    """
    # 1. Produit scalaire entre Q et K transposé sur ses dimensions
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # 2. Mise à l'échelle par la racine carrée de d_k
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 3. Application du masque optionnel (-1e9 ajouté aux éléments masqués)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 4. Softmax sur la dernière dimension (seq_len_v) pour obtenir les poids
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # 5. Multiplié par V pour obtenir la représentation pondérée
    output = tf.matmul(weights, V)

    return output, weights
