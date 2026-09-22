#!/usr/bin/env python3
"""Train a DQN agent to play Atari Breakout."""

import numpy as np
import gymnasium as gym
from gymnasium import Wrapper

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


class KerasRLGymWrapper(Wrapper):
    """Adapt a Gymnasium environment to the keras-rl API."""

    def reset(self, **kwargs):
        """Reset the environment and return only the observation."""
        observation, _ = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        """Run one step using the old Gym API expected by keras-rl."""
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )
        done = terminated or truncated
        return observation, reward, done, info

    def render(self, **kwargs):
        """Render the environment."""
        return self.env.render()


def build_model(input_shape, actions):
    """Build the convolutional neural network."""
    model = Sequential()

    model.add(
        Conv2D(
            32,
            (8, 8),
            strides=(4, 4),
            activation="relu",
            input_shape=input_shape
        )
    )
    model.add(
        Conv2D(
            64,
            (4, 4),
            strides=(2, 2),
            activation="relu"
        )
    )
    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu"
        )
    )
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(actions, activation="linear"))

    return model


def main():
    """Create the environment, train the agent and save the network."""

    env = gym.make("ALE/Breakout-v5")

    env = KerasRLGymWrapper(env)

    observation = env.reset()

    input_shape = observation.shape
    actions = env.action_space.n

    model = build_model(input_shape, actions)

    policy = EpsGreedyQPolicy()

    memory = SequentialMemory(
        limit=100000,
        window_length=4
    )

    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        nb_actions=actions,
        nb_steps_warmup=10000,
        target_model_update=10000,
        gamma=0.99
    )

    dqn.compile(
        Adam(learning_rate=0.00025),
        metrics=["mae"]
    )

    dqn.fit(
        env,
        nb_steps=100000,
        visualize=False,
        verbose=2
    )

    dqn.save_weights("policy.h5", overwrite=True)

    env.close()


if __name__ == "__main__":
    main()
