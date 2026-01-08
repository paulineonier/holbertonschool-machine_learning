#!/usr/bin/env python3
"""Module for Neural Style Transfer (NST) with model loading."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model


class NST:
    """
    Neural Style Transfer (NST) class.

    Attributes:
        style_layers (list of str): Layers used for style extraction.
        content_layer (str): Layer used for content extraction.
    """

    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initialize NST instance.

        Args:
            style_image (np.ndarray): Style reference image of shape (h, w, 3).
            content_image (np.ndarray): Content = image of shape (h, w, 3).
            alpha (float): Weight for content cost. Must be non-negative.
            beta (float): Weight for style cost. Must be non-negative.

        Raises:
            TypeError: If inputs are not of correct type or shape.
        """
        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3 \
                or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(content_image,
                          np.ndarray) or content_image.ndim != 3 \
                or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        # Charger le modèle VGG19 pour le style et le contenu
        self.model = self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescale an image such that its pixels are in [0, 1] and
        the largest side is 512 pixels.

        Args:
            image (np.ndarray): Input image of shape (h, w, 3).

        Returns:
            tf.Tensor: Scaled image of shape (1, h_new, w_new, 3).

        Raises:
            TypeError: If image is not a numpy.ndarray with shape (h, w, 3).
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3 \
                or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        scale = 512.0 / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize using bicubic interpolation
        image_resized = tf.image.resize(image,
                                        [new_h, new_w], method='bicubic')

        # Normalize pixel values to [0, 1]
        image_resized = tf.clip_by_value(image_resized / 255.0, 0.0, 1.0)

        # Add batch dimension
        image_resized = tf.expand_dims(image_resized, axis=0)
        return image_resized

    def load_model(self):
        """
        Load the VGG19 model configured for style and content extraction.

        Returns:
            tf.keras.Model: Keras model that outputs activations of
            style_layers followed by content_layer.
        """
        # Charger VGG19 pré-entraîné sur ImageNet
        vgg = VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False  # On ne veut pas entraîner le modèle

        # Obtenir les sorties pour les couches de style et contenu
        style_outputs = [vgg.get_layer(name).output
                         for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        outputs = style_outputs + [content_output]

        # Créer le modèle
        model = Model(inputs=vgg.input, outputs=outputs)
        return model
