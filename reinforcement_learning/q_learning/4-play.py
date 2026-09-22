#!/usr/bin/env python3
"""Play an episode using a trained Q-table."""

import numpy as np


def play(env, Q, max_steps=100):
    """Play an episode using the trained Q-table."""
    state = env.s
    total_rewards = 0
    rendered_outputs = []

    for _ in range(max_steps):
        rendered_outputs.append(env.render())

        action = np.argmax(Q[state])

        state, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward

        if terminated or truncated:
            rendered_outputs.append(env.render())
            break

    return total_rewards, rendered_outputs
