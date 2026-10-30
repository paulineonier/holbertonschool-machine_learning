#!/usr/bin/env python3
"""Module that trains an agent using Monte-Carlo Policy Gradient."""

import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Trains an agent using the Monte-Carlo Policy Gradient algorithm.

    Args:
        env: The initial environment.
        nb_episodes: Number of episodes used for training.
        alpha: The learning rate.
        gamma: The discount factor.

    Returns:
        A list containing the scores of all episodes.
    """
    # Initialize the weight matrix with random values
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    weights = np.random.rand(state_size, action_size)

    # Store the scores obtained during training
    scores = []

    # Loop through the requested number of episodes
    for episode in range(nb_episodes):

        # Reset the environment and initialize episode variables
        state, _ = env.reset()
        episode_rewards = []
        episode_gradients = []
        score = 0

        # Run the episode until termination or truncation
        while True:

            # Select an action and calculate its gradient
            action, gradient = policy_gradient(state, weights)

            # Execute the action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Store the reward and gradient
            episode_rewards.append(reward)
            episode_gradients.append(gradient)

            # Update the episode score
            score += reward

            # Move to the next state
            state = next_state

            # Stop if the episode has ended
            if terminated or truncated:
                break

        # Store the total score of the episode
        scores.append(score)

        # Initialize the discounted rewards array
        discounted_rewards = np.zeros(len(episode_rewards))

        # Calculate discounted rewards backwards
        cumulative_reward = 0
        for t in reversed(range(len(episode_rewards))):
            cumulative_reward = (
                episode_rewards[t] + gamma * cumulative_reward
            )
            discounted_rewards[t] = cumulative_reward

        # Update the weights using the policy gradients
        for gradient, reward in zip(
            episode_gradients, discounted_rewards
        ):
            weights += alpha * reward * gradient

        # Display the current episode number and its score
        print("Episode: {} Score: {}".format(episode, score))

    # Return all episode scores
    return scores
