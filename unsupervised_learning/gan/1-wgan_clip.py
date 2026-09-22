#!/usr/bin/env python3
"""
Module 1-wgan_clip
Contains the WGAN_clip class implementation for Wasserstein GAN
with weight clipping.
"""

import tensorflow as tf
from tensorflow import keras


class WGAN_clip(keras.Model):
    """
    Wasserstein GAN model with weight clipping.

    Attributes:
        latent_generator (function): Function generating latent vectors.
        real_examples (tf.Tensor): Dataset of real training examples.
        generator (keras.Model): Neural network generating fake samples.
        discriminator (keras.Model): Neural network evaluating sample validity.
        batch_size (int): Size of training batches.
        disc_iter (int): Discriminator steps per generator step.
        learning_rate (float): Optimizer learning rate.
        beta_1 (float): Adam optimizer exponential decay rate for 1st moment.
        beta_2 (float): Adam optimizer exponential decay rate for 2nd moment.
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """
        Initializes WGAN_clip with Wasserstein loss functions and optimizers.

        Args:
            generator (keras.Model): The generator network.
            discriminator (keras.Model): The discriminator network.
            latent_generator (function): Function to sample latent vectors.
            real_examples (tf.Tensor): Dataset of real training instances.
            batch_size (int, optional): Size of training batch. Default 200.
            disc_iter (int, optional): Discriminator updates per step.
                                       Default 2.
            learning_rate (float, optional): Optimizer learning rate.
                                             Default 0.005.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # Generator loss: -E[D(G(z))]
        self.generator.loss = lambda x: -tf.math.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=generator.optimizer,
            loss=generator.loss
        )

        # Discriminator loss: E[D(G(z))] - E[D(x)]
        self.discriminator.loss = lambda x, y: (
            tf.math.reduce_mean(x) - tf.math.reduce_mean(y)
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.discriminator.compile(
            optimizer=discriminator.optimizer,
            loss=discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """
        Generates a batch of fake samples using the generator.

        Args:
            size (int, optional): Number of fake samples. Defaults to
                self.batch_size.
            training (bool, optional): Whether training mode is active.
                Defaults to False.

        Returns:
            tf.Tensor: Generated fake samples.
        """
        if not size:
            size = self.batch_size
        return self.generator(
            self.latent_generator(size),
            training=training
        )

    def get_real_sample(self, size=None):
        """
        Extracts a random subset of real training samples.

        Args:
            size (int, optional): Number of real samples. Defaults to
                self.batch_size.

        Returns:
            tf.Tensor: Random subset of real samples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """
        Executes one training step of WGAN with discriminator weight clipping.

        Args:
            useless_argument: Required placeholder by Keras fit API.

        Returns:
            dict: Dictionary containing 'discr_loss' and 'gen_loss'.
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)

                pred_real = self.discriminator(real_samples, training=True)
                pred_fake = self.discriminator(fake_samples, training=True)

                discr_loss = self.discriminator.loss(pred_fake, pred_real)

            gradients_of_discriminator = disc_tape.gradient(
                discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(
                    gradients_of_discriminator,
                    self.discriminator.trainable_variables
                )
            )

            # Weight clipping between -1 and 1
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        with tf.GradientTape() as gen_tape:
            fake_samples = self.get_fake_sample(training=True)
            pred_fake = self.discriminator(fake_samples, training=True)

            gen_loss = self.generator.loss(pred_fake)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        self.generator.optimizer.apply_gradients(
            zip(
                gradients_of_generator,
                self.generator.trainable_variables
            )
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
