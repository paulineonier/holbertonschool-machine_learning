#!/usr/bin/env python3
"""Module d'analyse de texte pour l'extraction de bag of words."""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Crée une matrice d'embeddings de type bag of words à partir de phrases.

    Parameters:
    sentences (list): liste de chaînes de caractères (phrases).
    vocab (list, optional): liste des mots du vocabulaire à utiliser.

    Returns:
    tuple: (embeddings, features)
        - embeddings: numpy.ndarray de forme (s, f) avec le décompte des mots.
        - features: liste des mots clés utilisés pour les colonnes.
    """
    cleaned_sentences = []

    for s in sentences:
        # Nettoyage : suppression du 's possessif et de la ponctuation
        s_clean = re.sub(r"'s\b", "", s.lower())
        tokens = re.findall(r"\b\w+\b", s_clean)
        cleaned_sentences.append(tokens)

    # Définition des features
    if vocab is None:
        features = set()
        for tokens in cleaned_sentences:
            features.update(tokens)
        features = sorted(list(features))
    else:
        features = vocab

    # Construction de la matrice
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, tokens in enumerate(cleaned_sentences):
        for token in tokens:
            if token in features:
                j = features.index(token)
                embeddings[i, j] += 1

    return embeddings, features