#!/usr/bin/env python3
"""Performs SARSA(lambda)."""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """Perform SARSA(lambda) using eligibility traces."""
    n_states, n_actions = Q.shape

    for _ in range(episodes):
        eligibility = np.zeros_like(Q)
        state, _ = env.reset()

        # Choose the initial action using epsilon-greedy
        if np.random.uniform() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Choose the next action
            if np.random.uniform() < epsilon:
                next_action = np.random.randint(n_actions)
            else:
                next_action = np.argmax(Q[next_state])

            # Compute the TD error
            next_value = 0 if terminated else Q[next_state, next_action]
            delta = reward + gamma * next_value - Q[state, action]

            # Update eligibility trace for the current state-action pair
            eligibility[state, action] += 1

            # Update the Q table
            Q += alpha * delta * eligibility

            # Decay eligibility traces
            eligibility *= gamma * lambtha

            if terminated or truncated:
                break

            state = next_state
            action = next_action

        # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q
