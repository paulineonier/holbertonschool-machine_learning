#!/usr/bin/env python3
"""Module pour le calcul du score BLEU unigramme."""
import numpy as np


def uni_bleu(references, sentence):
    """Calcule le score BLEU unigramme pour une phrase proposée.

    Parameters:
    references (list): liste des traductions de référence.
                       chaque référence est une liste de mots.
    sentence (list): liste des mots de la phrase proposée par le modèle.

    Returns:
    float: le score BLEU unigramme.
    """
    c = len(sentence)
    if c == 0:
        return 0.0

    # 1. Calcul des Clipped Counts pour les unigrammes
    words_count = {}
    for word in sentence:
        words_count[word] = words_count.get(word, 0) + 1

    clipped_count = 0
    for word, count in words_count.items():
        max_ref_count = 0
        for ref in references:
            max_ref_count = max(max_ref_count, ref.count(word))
        clipped_count += min(count, max_ref_count)

    precision = clipped_count / c

    # 2. Recherche de la longueur de référence la plus proche (r)
    ref_lens = [len(ref) for ref in references]
    # Tri par différence absolue, & par la longueur minimale en cas d'égalité
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))

    # 3. Calcul de la pénalité de brièveté (Brevity Penalty - BP)
    if c > r:
        bp = 1.0
    else:
        bp = np.exp(1 - r / c)

    return float(bp * precision)
