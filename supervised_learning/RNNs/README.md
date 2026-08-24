# Recurrent Neural Networks (RNNs)

## Overview

This project focuses on the foundational architectures used for processing sequential and temporal data. It covers step-by-step implementations of basic **Recurrent Neural Networks (RNNs)**, **Gated Recurrent Units (GRUs)**, **Long Short-Term Memory networks (LSTMs)**, and **Bidirectional RNNs (BRNNs)** using pure **NumPy** without relying on high-level deep learning frameworks.

---

## What I Am Going to Implement

In this project, I will build modular Python classes and functions from scratch to model sequence processing and forward/backward propagation:

1. **Simple RNN Cell & Forward Pass**
   * Implement `RNNCell` to perform one-step forward propagation using concatenated hidden states and input vectors.
   * Build the `rnn` function to process an entire temporal sequence over multiple time steps.

2. **Gated Recurrent Unit (GRU) Cell & Forward Pass**
   * Implement `GRUCell` featuring update gates ($z$) and reset gates ($r$) to control information flow and mitigate vanishing gradients.
   * Implement sequence-wide forward propagation through chained GRU cells.

3. **Long Short-Term Memory (LSTM) Cell & Forward Pass**
   * Implement `LSTMCell` managing cell state ($c$) and hidden state ($h$) via forget gates ($f$), update/input gates ($i$), and output gates ($o$).
   * Build the complete forward sequence processing for LSTMs.

4. **Bidirectional RNN (BRNN)**
   * Implement bidirectional sequence processing running forward and backward temporal passes concurrently, concatenating hidden representations at each time step.

5. **Backpropagation Through Time (BPTT)**
   * Implement gradient computation and backward passes for simple RNN cells and full sequences to understand gradient updates and sequence learning dynamics.

---

## Technical Concepts Covered

* **Sequence Data Modeling:** Processing variable-length input sequences $X = (x_1, x_2, \dots, x_T)$.
* **Vanishing & Exploding Gradient Problems:** How long-term temporal dependencies lead to numerical instability during BPTT.
* **Gating Mechanisms:** How sigmoid-gated memory units in LSTMs and GRUs regulate information retention and additive gradient flow.
* **Bidirectional Context:** Leveraging past and future context simultaneously for sequence labeling and feature extraction.

---

## Project Requirements & Constraints

* **Environment:** Executed on Ubuntu 20.04 LTS using Python 3.9.
* **Libraries:** Strictly restricted to `import numpy as np`.
* **Code Quality:**
  * Conform strictly to `pycodestyle` (v2.11.1).
  * Include comprehensive docstrings for all modules, classes, and functions.
  * Executable files starting with `#!/usr/bin/env python3` and ending with a newline.

---

## Author
* Project developed as part of the Specialization curriculum in Machine Learning & Deep Learning.