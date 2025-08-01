#!/usr/bin/env python3
"""
Function that updates the learning rate using inverse decay time
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay.

    Parameters:
    - alpha: initial learning rate (float)
    - decay_rate: decay rate (float)
    - global_step: number of steps completed (int)
    - decay_step: step interval before applying decay (int)

    Returns:
    - Updated learning rate (float)
    """
    decay_factor = np.floor(global_step / decay_step)
    alpha_updated = alpha / (1 + decay_rate * decay_factor)
    return alpha_updated
