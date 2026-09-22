#!/usr/bin/env python3
"""
Module 4-wgan_gp
Contains the WGAN_GP class implementation extended with weight loading
capabilities for pre-trained generator and discriminator models.
"""

import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """
    Wasserstein GAN model with Gradient Penalty (WGAN-GP).

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
        lambda_gp (float): Weight factor for the gradient penalty term.
        dims (tf.TensorShape): Shape of the real training examples.
        len_dims (tf.Tensor): Number of dimensions in real_examples.
        axis (tf.Tensor): Axes over which to calculate the gradient norm.
        scal_shape (tf.Tensor): Target shape for uniform interpolation tensor.
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        """
        Initializes WGAN_GP with Wasserstein losses, optimizers, and GP setup.

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
            lambda_gp (float, optional): Gradient penalty parameter.
                                         Default 10.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .3
        self.beta_2 = .9

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

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

    def replace_weights(self, gen_h5, disc_h5):
        """
        Replaces generator and discriminator weights with pre-trained weights.

        Args:
            gen_h5 (str): Path to the HDF5 (.h5) file with generator weights.
            disc_h5 (str): Path to the HDF5 (.h5) file with discriminator
                weights.
        """
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)

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

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Computes an interpolated sample between real and fake samples.

        Args:
            real_sample (tf.Tensor): Batch of real training samples.
            fake_sample (tf.Tensor): Batch of generated fake samples.

        Returns:
            tf.Tensor: Convex combination of real and fake samples.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Calculates the gradient penalty over the interpolated sample.

        Args:
            interpolated_sample (tf.Tensor): Interpolated samples.

        Returns:
            tf.Tensor: Scalar value representing the gradient penalty.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """
        Executes one training step of WGAN-GP (Discriminator + Generator).

        Args:
            useless_argument: Required placeholder by Keras fit API.

        Returns:
            dict: Dictionary containing 'discr_loss', 'gen_loss', and 'gp'.
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)
                interpolated_samples = self.get_interpolated_sample(
                    real_samples, fake_samples
                )

                pred_real = self.discriminator(real_samples, training=True)
                pred_fake = self.discriminator(fake_samples, training=True)

                discr_loss = self.discriminator.loss(pred_fake, pred_real)
                gp = self.gradient_penalty(interpolated_samples)
                new_discr_loss = discr_loss + self.lambda_gp * gp

            gradients_of_discriminator = disc_tape.gradient(
                new_discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(
                    gradients_of_discriminator,
                    self.discriminator.trainable_variables
                )
            )

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

        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
