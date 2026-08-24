# Autoencoders and Variational Autoencoders (VAEs)

## Overview

This project explores **unsupervised learning** and **generative modeling** using **Deep Autoencoders** and **Variational Autoencoders (VAEs)** built with **TensorFlow 2.15 / Keras**. 

The main goal of this repository is to implement, train, and evaluate various autoencoder architectures—ranging from basic linear compression models to deep, convolutional, sparse, and generative architectures capable of generating novel synthetic data.

---

## What I Am Going to Implement

Throughout this project, I will develop Python modules covering the following architectures and tasks:

1. **Vanilla / Basic Autoencoder**
   * Build a standard dense autoencoder that compresses high-dimensional input into a lower-dimensional latent bottleneck and reconstructs it.
   * Implement custom loss functions to evaluate reconstruction fidelity (e.g., Mean Squared Error or Binary Cross-Entropy).

2. **Deep Autoencoder**
   * Stack multiple hidden layers in both the encoder and decoder to capture hierarchical abstract representations of complex datasets.

3. **Sparse Autoencoder**
   * Introduce L1 regularization or KL divergence sparsity constraints on the latent activations, forcing the network to learn efficient, sparse features.

4. **Convolutional Autoencoder**
   * Implement 2D spatial feature extraction using `Conv2D` and `Conv2DTranspose` layers, tailored specifically for image reconstruction and denoising tasks.

5. **Variational Autoencoder (VAE)**
   * Build a probabilistic generative model where the encoder maps inputs into latent distribution parameters (mean $\mu$ and log-variance $\log(\sigma^2)$).
   * Implement the **reparameterization trick** ($z = \mu + \sigma \odot \epsilon$) to enable gradient propagation through stochastic nodes.
   * Formulate the composite VAE loss function combining **Reconstruction Loss** and **Kullback-Leibler (KL) Divergence** to regularize the latent space towards a standard normal distribution $\mathcal{N}(0, I)$.
   * Sample from the learned latent manifold to generate entirely new synthetic data samples.

---

## Technical Objectives & Concepts Covered

* **Latent Space & Bottleneck:** Understanding how dimensionality reduction forces network compression and feature extraction.
* **Kullback-Leibler (KL) Divergence:** Measuring information loss and probability distribution divergence in probabilistic models.
* **Generative vs. Discriminative Models:** Transitioning from passive representations to active sampling and content generation.
* **Latent Space Interpolation:** Visualizing smooth transitions and structure within low-dimensional representations.

---

## Project Constraints & Requirements

* **Environment:** Executed on Ubuntu 20.04 LTS using Python 3.9.
* **Libraries:** `numpy` (v1.25.2) and `tensorflow` (v2.15).
* **Allowed Imports:** Only `import tensorflow.keras as keras` (unless specified otherwise).
* **Code Quality:** 
  * Strict adherence to `pycodestyle` (v2.11.1).
  * Comprehensive docstrings for all modules, classes, and functions.
  * Executable files starting with `#!/usr/bin/env python3` and ending with a newline.

---

## Author
* Project developed as part of the Specialization curriculum in Machine Learning & Deep Learning.