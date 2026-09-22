#!/usr/bin/env python3
"""Play Atari Breakout using a trained DQN agent."""

import gymnasium as gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy


class KerasRLGymWrapper(gym.Wrapper):
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
    """Build the same neural network used during training."""
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
    """Load the trained agent and play Breakout."""

    env = gym.make(
        "ALE/Breakout-v5",
        render_mode="human"
    )

    env = KerasRLGymWrapper(env)

    observation = env.reset()

    input_shape = observation.shape
    actions = env.action_space.n

    model = build_model(input_shape, actions)

    memory = SequentialMemory(
        limit=100000,
        window_length=4
    )

    policy = GreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        nb_actions=actions,
        nb_steps_warmup=0,
        gamma=0.99
    )

    dqn.compile("adam")

    dqn.load_weights("policy.h5")

    dqn.test(
        env,
        nb_episodes=1,
        visualize=True
    )

    env.close()


if __name__ == "__main__":
    main()
