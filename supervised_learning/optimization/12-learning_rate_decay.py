#!/usr/bin/env python3
"""
Function that creates a learning rate decay operation in tensorflow
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates an inverse time decay learning rate schedule in TensorFlow.

    Parameters:
    - alpha: initial learning rate (float)
    - decay_rate: rate at which the learning rate decays (float)
    - decay_step: number of steps before applying each decay (int)

    Returns:
    - A tf.keras.optimizers.schedules.LearningRateSchedule object
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
