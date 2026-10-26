#!/usr/bin/env python3
"""Performs the Monte Carlo algorithm."""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Update the value estimate using Monte Carlo."""
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))

            if terminated or truncated:
                break

            state = next_state

        G = 0
        for state, reward in reversed(episode):
            G = reward + gamma * G
            V[state] += alpha * (G - V[state])

    return V
