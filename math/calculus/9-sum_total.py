"""
Module : summation
Module contient une fonction qui calcule la somme des carrés de tous les
entiers de 1 à n.
"""


def summation_i_squared(n):
    """
    Calcule la somme des carrés de tous les entiers de 1 à n.

    Args:
        n (int): Le nombre entier jusqu'auquel calculer la somme des carrés.

    Returns:
        int ou None: La somme des carrés si l'entrée est valide, sinon None.

    Notes:
        La formule utilisée est :
        somme des carrés = (n * (n + 1) * (2n + 1)) // 6
    """
    # Vérification de l'entrée
    if n is None or n < 0:
        return None

    # Calcul avec la formule mathématique
    return (n * (n + 1) * (2 * n + 1)) // 6
