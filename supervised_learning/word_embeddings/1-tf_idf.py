#!/usr/bin/env python3
"""Module d'analyse de texte pour l'extraction TF-IDF."""
import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """Crée une matrice d'embeddings TF-IDF à partir de phrases.

    Parameters:
    sentences (list): liste de chaînes de caractères (phrases).
    vocab (list, optional): liste des mots du vocabulaire à utiliser.

    Returns:
    tuple: (embeddings, features)
        - embeddings: numpy.ndarray de forme (s, f) contenant les TF-IDF.
        - features: liste des mots clés utilisés pour les colonnes.
    """
    # Étape 1 : Tokenisation et nettoyage des phrases
    cleaned_sentences = []
    for s in sentences:
        tokens = re.findall(r"\b[a-zA-Z0-9]+\b", s.lower())
        cleaned_sentences.append(tokens)

    # Étape 2 : Définition des features
    if vocab is None:
        features = set()
        for tokens in cleaned_sentences:
            features.update(tokens)
        features = sorted(list(features))
    else:
        features = vocab

    s = len(sentences)
    f = len(features)

    # Étape 3 : Calcul du Term Frequency (TF)
    tf = np.zeros((s, f))
    for i, tokens in enumerate(cleaned_sentences):
        for token in tokens:
            if token in features:
                j = features.index(token)
                tf[i, j] += 1

    # Étape 4 : Calcul du Document Frequency (DF) et Inverse Document Frequency (IDF)
    # Formule lissée standard (type scikit-learn): log((1 + N) / (1 + df)) + 1
    df = np.count_nonzero(tf > 0, axis=0)
    idf = np.log((1 + s) / (1 + df)) + 1

    # Étape 5 : Produit TF * IDF
    tf_idf_matrix = tf * idf

    # Étape 6 : Normalisation L2 par ligne (L2 norm)
    # Évite la division par zéro pour les phrases sans aucun mot du vocabulaire
    row_norms = np.linalg.norm(tf_idf_matrix, axis=1, keepdims=True)
    embeddings = np.zeros_like(tf_idf_matrix)
    
    mask = row_norms.flatten() > 0
    embeddings[mask] = tf_idf_matrix[mask] / row_norms[mask]

    return embeddings, features