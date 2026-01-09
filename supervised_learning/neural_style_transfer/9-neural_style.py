#!/usr/bin/env python3
"""
Neural Style Transfer
"""

import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class"""

    def __init__(self, style_image, content_image,
                 alpha=1e4, beta=1):
        """
        Class constructor
        """

        # ---------- Type checks ----------

        if not isinstance(style_image, np.ndarray) or \
           style_image.ndim != 3:
            raise TypeError
        ("style_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray) or \
           content_image.ndim != 3:
            raise TypeError
        ("content_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a number")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        if not isinstance(beta, (int, float)):
            raise TypeError("beta must be a number")
        if beta < 0:
            raise ValueError("beta must be positive")

        self.alpha = alpha
        self.beta = beta

        # ---------- Image preprocessing ----------

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

        # ---------- Load VGG19 ----------

        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        self.style_layers = [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
            'block5_conv1'
        ]
        self.content_layer = 'block5_conv2'

        outputs = ([vgg.get_layer(name).output
                    for name in self.style_layers] +
                   [vgg.get_layer(self.content_layer).output])

        self.model = tf.keras.Model(vgg.input, outputs)

        # ---------- Extract features ----------

        style_outputs = self.model(self.style_image)
        content_outputs = self.model(self.content_image)

        self.gram_style_features = [
            self.gram_matrix(output)
            for output in style_outputs[:-1]
        ]

        self.content_feature = content_outputs[-1]

    @staticmethod
    def scale_image(image):
        """Rescales image to [0,1] and adds batch dimension"""

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = image / 255.0
        image = tf.expand_dims(image, axis=0)
        return image

    @staticmethod
    def gram_matrix(input_layer):
        """Computes Gram matrix"""

        _, h, w, c = input_layer.shape
        features = tf.reshape(input_layer, (h * w, c))
        gram = tf.matmul(features, features, transpose_a=True)
        return gram / tf.cast(h * w, tf.float32)

    def compute_content_cost(self, content, generated):
        """Computes content cost"""

        return tf.reduce_mean(tf.square(generated - content))

    def compute_style_cost(self, style, generated):
        """Computes style cost"""

        return tf.reduce_mean(tf.square(generated - style))

    def compute_cost(self, generated_image):
        """Computes total cost"""

        outputs = self.model(generated_image)
        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        # Content cost
        J_content = self.compute_content_cost(
            self.content_feature,
            content_output
        )

        # Style cost
        J_style = 0
        for gen, gram in zip(style_outputs, self.gram_style_features):
            J_style += self.compute_style_cost(
                self.gram_matrix(gen),
                gram
            )
        J_style /= len(self.style_layers)

        # Total cost
        J_total = self.alpha * J_content + self.beta * J_style

        return J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None,
                       lr=0.01, beta1=0.9, beta2=0.99):
        """Generates stylized image"""

        # ---------- Checks ----------

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError(
                    "step must be positive and less than iterations"
                )

        if not isinstance(lr, (int, float)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")

        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")

        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        # ---------- Initialization ----------

        generated_image = tf.Variable(self.content_image, dtype=tf.float32)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2
        )

        best_cost = float('inf')
        best_image = None

        # ---------- Training loop ----------

        for i in range(iterations + 1):

            with tf.GradientTape() as tape:
                J_total, J_content, J_style = self.compute_cost(
                    generated_image
                )

            grads = tape.gradient(J_total, generated_image)
            optimizer.apply_gradients([(grads, generated_image)])

            generated_image.assign(
                tf.clip_by_value(generated_image, 0.0, 1.0)
            )

            if J_total < best_cost:
                best_cost = J_total
                best_image = tf.identity(generated_image)

            if step is not None and i % step == 0:
                print(
                    f"Cost at iteration {i}: {J_total.numpy()}, "
                    f"content {J_content.numpy()}, "
                    f"style {J_style.numpy()}"
                )

        return best_image.numpy(), best_cost.numpy()
