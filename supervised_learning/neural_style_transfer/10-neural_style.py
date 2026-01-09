#!/usr/bin/env python3
"""
Neural Style Transfer with Variational Cost
"""

import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class"""

    def __init__(self, style_image, content_image,
                 alpha=1e4, beta=1, var=10):
        """Class constructor"""

        # ---------- Checks ----------

        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(content_image,
                          np.ndarray) or content_image.ndim != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        if not isinstance(var, (int, float)) or var < 0:
            raise TypeError("var must be a non-negative number")

        self.alpha = alpha
        self.beta = beta
        self.var = var

        # ---------- Preprocess images ----------

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
        """Scales image to [0,1] and adds batch dimension"""
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = image / 255.0
        return tf.expand_dims(image, axis=0)

    @staticmethod
    def gram_matrix(input_layer):
        """Computes Gram matrix"""
        _, h, w, c = input_layer.shape
        features = tf.reshape(input_layer, (h * w, c))
        gram = tf.matmul(features, features, transpose_a=True)
        return gram / tf.cast(h * w, tf.float32)

    @staticmethod
    def variational_cost(generated_image):
        """
        Computes variational cost (total variation)
        """
        dh = generated_image[:, 1:, :, :] - generated_image[:, :-1, :, :]
        dw = generated_image[:, :, 1:, :] - generated_image[:, :, :-1, :]
        return tf.reduce_sum(tf.square(dh)) + tf.reduce_sum(tf.square(dw))

    def compute_content_cost(self, content, generated):
        """Content cost"""
        return tf.reduce_mean(tf.square(generated - content))

    def compute_style_cost(self, style, generated):
        """Style cost"""
        return tf.reduce_mean(tf.square(generated - style))

    def total_cost(self, generated_image):
        """
        Computes total cost
        Returns: J, J_content, J_style, J_var
        """

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

        # Variational cost
        J_var = self.variational_cost(generated_image)

        # Total cost
        J = (self.alpha * J_content +
             self.beta * J_style +
             self.var * J_var)

        return J, J_content, J_style, J_var

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
                    "iterations must be positive and less than iterations"
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
                J, Jc, Js, Jv = self.total_cost(generated_image)

            grads = tape.gradient(J, generated_image)
            optimizer.apply_gradients([(grads, generated_image)])

            generated_image.assign(
                tf.clip_by_value(generated_image, 0.0, 1.0)
            )

            if J < best_cost:
                best_cost = J
                best_image = tf.identity(generated_image)

            if step is not None and i % step == 0:
                print(
                    f"Cost at iteration {i}: {J.numpy()}, "
                    f"content {Jc.numpy()}, "
                    f"style {Js.numpy()}, "
                    f"var {Jv.numpy()}"
                )

        return best_image.numpy(), best_cost.numpy()
