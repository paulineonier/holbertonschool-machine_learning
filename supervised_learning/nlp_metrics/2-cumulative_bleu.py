#!/usr/bin/env python3
"""Module pour le calcul du score BLEU cumulé."""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """Calcule le score BLEU cumulé n-gramme pour une phrase proposée.

    Parameters:
    references (list): liste des traductions de référence (listes de mots).
    sentence (list): liste des mots de la phrase proposée par le modèle.
    n (int): la taille maximale des n-grammes à évaluer.

    Returns:
    float: le score BLEU cumulé jusqu'à n.
    """
    c = len(sentence)
    if c == 0:
        return 0.0

    # Poids uniformes pour chaque ordre de n-gramme (1 à n)
    weights = [1.0 / n] * n
    precisions = []

    # 1. Calcul de la précision tronquée pour chaque k-gramme (de k = 1 à n)
    for k in range(1, n + 1):
        if c < k:
            precisions.append(0.0)
            continue

        sentence_ngrams = {}
        for i in range(c - k + 1):
            ngram = tuple(sentence[i:i + k])
            sentence_ngrams[ngram] = sentence_ngrams.get(ngram, 0) + 1

        clipped_count = 0
        total_ngrams = sum(sentence_ngrams.values())

        for ngram, count in sentence_ngrams.items():
            max_ref_count = 0
            for ref in references:
                ref_len = len(ref)
                ref_ngram_count = 0
                for i in range(ref_len - k + 1):
                    if tuple(ref[i:i + k]) == ngram:
                        ref_ngram_count += 1
                max_ref_count = max(max_ref_count, ref_ngram_count)
            clipped_count += min(count, max_ref_count)

        precision_k = clipped_count / total_ngrams
        precisions.append(precision_k)

    # 2. Recherche de la longueur de référence la plus proche (r)
    ref_lens = [len(ref) for ref in references]
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))

    # 3. Calcul de la pénalité de brièveté (Brevity Penalty - BP)
    if c > r:
        bp = 1.0
    else:
        bp = np.exp(1 - r / c)

    # 4. Combinaison géométrique des précisions
    # Si une des précisions est égale à 0, le score cumulé devient 0
    if any(p == 0 for p in precisions):
        return 0.0

    s_precisions = sum(w * np.log(p) for w, p in zip(weights, precisions))
    score = bp * np.exp(s_precisions)

    return float(score)
