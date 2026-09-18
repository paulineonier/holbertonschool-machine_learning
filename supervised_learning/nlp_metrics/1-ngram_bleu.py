#!/usr/bin/env python3
"""Module pour le calcul du score BLEU n-gramme."""
import numpy as np


def ngram_bleu(references, sentence, n):
    """Calcule le score BLEU n-gramme pour une phrase proposée.

    Parameters:
    references (list): liste des traductions de référence (listes de mots).
    sentence (list): liste des mots de la phrase proposée.
    n (int): la taille des n-grammes à évaluer.

    Returns:
    float: le score BLEU n-gramme.
    """
    c = len(sentence)
    if c < n:
        return 0.0

    # 1. Extraction des n-grammes de la phrase proposée
    sentence_ngrams = {}
    for i in range(c - n + 1):
        ngram = tuple(sentence[i:i + n])
        sentence_ngrams[ngram] = sentence_ngrams.get(ngram, 0) + 1

    # 2. Calcul des occurrences tronquées (Clipped Counts) dans les références
    clipped_count = 0
    total_ngrams = sum(sentence_ngrams.values())

    for ngram, count in sentence_ngrams.items():
        max_ref_count = 0
        for ref in references:
            ref_len = len(ref)
            ref_ngram_count = 0
            for i in range(ref_len - n + 1):
                if tuple(ref[i:i + n]) == ngram:
                    ref_ngram_count += 1
            max_ref_count = max(max_ref_count, ref_ngram_count)
        clipped_count += min(count, max_ref_count)

    precision = clipped_count / total_ngrams

    # 3. Sélection de la longueur de référence la plus proche (r)
    ref_lens = [len(ref) for ref in references]
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))

    # 4. Pénalité de brièveté (BP)
    if c > r:
        bp = 1.0
    else:
        bp = np.exp(1 - r / c)

    return float(bp * precision)
