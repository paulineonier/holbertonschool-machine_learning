#!/usr/bin/env python3
"""
Neural Style Transfer (NST) implementation.
"""

import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer (NST) class."""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initialize the NST object.

        Args:
            style_image (np.ndarray): style reference image (h, w, 3)
            content_image (np.ndarray): content reference image (h, w, 3)
            alpha (float): content cost weight
            beta (float): style cost weight
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

        self.model = self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Scale image to range [0, 1] and max dimension 512."""
        if not isinstance(image, np.ndarray) or image.ndim != 3 \
                or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        scale = 512 / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        image = tf.image.resize(
            image, [new_h, new_w], method='bicubic'
        )
        image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
        return tf.expand_dims(image, axis=0)

    def load_model(self):
        """Load VGG19 model for NST."""
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        style_outputs = [
            vgg.get_layer(name).output for name in self.style_layers
        ]
        content_output = vgg.get_layer(self.content_layer).output

        return tf.keras.Model(
            inputs=vgg.input,
            outputs=style_outputs + [content_output]
        )

    @staticmethod
    def gram_matrix(input_layer):
        """Compute Gram matrix from tensor."""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) \
                or len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape
        features = tf.reshape(input_layer, (h * w, c))
        gram = tf.matmul(features, features, transpose_a=True)
        gram /= tf.cast(h * w, tf.float32)
        return tf.expand_dims(gram, axis=0)

    def generate_features(self):
        """Extract style and content features."""
        vgg19 = tf.keras.applications.vgg19
        style_input = vgg19.preprocess_input(self.style_image * 255)
        content_input = vgg19.preprocess_input(self.content_image * 255)

        style_outputs = self.model(style_input)[:-1]
        content_output = self.model(content_input)[-1]

        self.gram_style_features = [
            self.gram_matrix(output) for output in style_outputs
        ]
        self.content_feature = content_output

    def layer_style_cost(self, style_output, gram_target):
        """Compute style cost for a single layer."""
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) \
                or len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        _, _, _, c = style_output.shape
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) \
                or gram_target.shape != (1, c, c):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c
                )
            )

        gram_generated = self.gram_matrix(style_output)
        return tf.reduce_sum(
            tf.square(gram_generated - gram_target)
        ) / (c ** 2)

    def style_cost(self, style_outputs):
        """Compute total style cost."""
        if not isinstance(style_outputs, list) \
                or len(style_outputs) != len(self.style_layers):
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    len(self.style_layers)
                )
            )

        weight = 1 / len(self.style_layers)
        cost = 0.0

        for i, output in enumerate(style_outputs):
            cost += weight * self.layer_style_cost(
                output, self.gram_style_features[i]
            )
        return cost

    def content_cost(self, content_output):
        """Compute content cost."""
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) \
                or content_output.shape != self.content_feature.shape:
            raise TypeError(
                "content_output must be a tensor of shape {}".format(
                    self.content_feature.shape
                )
            )

        _, h, w, c = self.content_feature.shape
        cost = tf.reduce_sum(
            tf.square(content_output - self.content_feature)
        )
        return cost / tf.cast(h * w * c, tf.float32)

    def total_cost(self, generated_image):
        """Compute total cost."""
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) \
                or generated_image.shape != self.content_image.shape:
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    self.content_image.shape
                )
            )

        vgg19 = tf.keras.applications.vgg19
        preprocessed = vgg19.preprocess_input(generated_image * 255)
        outputs = self.model(preprocessed)

        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        j_content = self.content_cost(content_output)
        j_style = self.style_cost(style_outputs)
        j_total = self.alpha * j_content + self.beta * j_style

        return j_total, j_content, j_style
