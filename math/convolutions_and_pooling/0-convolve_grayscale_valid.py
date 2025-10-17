#!/usr/bin/env python3
import numpy as np

def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images
    
    Args:
        images: numpy.ndarray of shape (m, h, w)
            containing multiple grayscale images
        kernel: numpy.ndarray of shape (kh, kw)
            containing the kernel for the convolution
    
    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # dimensions de sortie
    new_h = h - kh + 1
    new_w = w - kw + 1

    # tableau de sortie
    output = np.zeros((m, new_h, new_w))

    # convolution avec deux boucles (sur hauteur et largeur)
    for i in range(new_h):
        for j in range(new_w):
            # sélection du patch de taille (m, kh, kw)
            patch = images[:, i:i+kh, j:j+kw]
            # produit élément par élément et somme
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
