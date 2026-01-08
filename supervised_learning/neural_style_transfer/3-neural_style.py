#!/usr/bin/env python3
"""
Neural Style Transfer (NST) class with feature extraction.
"""

import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer (NST) class."""

    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize NST instance."""
        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3 \
                or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if not isinstance(content_image, np.ndarray) or content_image.ndim != 3 \
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
        self.model = self.load_model()
        # Do not generate features automatically to keep 0-main.py compatible
        self.gram_style_features = None
        self.content_feature = None

    @staticmethod
    def scale_image(image):
        """Rescale image to [0,1] with max side 512."""
        if not isinstance(image, np.ndarray) or image.ndim != 3 \
                or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )
        h, w, _ = image.shape
        scale = 512.0 / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        image_resized = tf.image.resize(image, [new_h, new_w],
                                        method='bicubic')
        image_resized = tf.clip_by_value(image_resized / 255.0, 0.0, 1.0)
        return tf.expand_dims(image_resized, axis=0)

    def load_model(self):
        """Load VGG19 model for style and content extraction."""
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        vgg.trainable = False
        style_outputs = [vgg.get_layer(name).output
                         for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        outputs = style_outputs + [content_output]
        return tf.keras.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Compute Gram matrix of a tensor of shape (1, h, w, c)."""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) \
                or len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        batch, h, w, c = input_layer.shape
        features = tf.reshape(input_layer, (h * w, c))
        gram = tf.linalg.matmul(tf.transpose(features), features)
        gram = gram / tf.cast(h * w, tf.float32)
        return tf.expand_dims(gram, axis=0)

    def generate_features(self):
        """
        Extract style and content features from images.

        Sets:
            gram_style_features (list): list of Gram matrices
            content_feature (tf.Tensor): content layer output
        """
        outputs = self.model(tf.concat([self.style_image,
                                        self.content_image], axis=0))
        style_outputs = outputs[:-1]  # style layers
        content_output = outputs[-1]   # content layer

        self.gram_style_features = [
            self.gram_matrix(layer[0:1]) for layer in style_outputs
        ]
        self.content_feature = content_output[1:2]
