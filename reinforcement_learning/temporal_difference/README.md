# Reinforcement Learning - Temporal Difference

## Description

This project introduces Temporal Difference (TD) learning algorithms in Reinforcement Learning. It focuses on learning state values and state-action values through interactions with an environment.

The algorithms are tested using Gymnasium and the FrozenLake environment.

## Learning Objectives

At the end of this project, you should be able to:

-   Explain the concepts of Temporal Difference learning.
    
-   Understand the Monte Carlo algorithm.
    
-   Implement TD(λ) using eligibility traces.
    
-   Understand and implement the SARSA(λ) algorithm.
    
-   Explain the difference between state-value functions and action-value functions.
    
-   Use an epsilon-greedy strategy to balance exploration and exploitation.
    
-   Update value estimates using rewards, discount factors, and learning rates.
    

## Requirements

-   Ubuntu 20.04 LTS
    
-   Python 3.9
    
-   NumPy
    
-   Gymnasium
    
-   pycodestyle 2.11.1
    

## Project Structure

```
temporal_difference/
├── 0-monte_carlo.py
├── 1-td_lambtha.py
├── 2-sarsa_lambtha.py
├── 0-main.py
├── 1-main.py
├── 2-main.py
└── README.md
```

## Algorithms

### 0\. Monte Carlo

**File:** `0-monte_carlo.py`

Monte Carlo estimates the value of states by generating episodes and calculating the discounted return after each episode.

The value estimate is updated using the learning rate `alpha`.

### 1\. TD(λ)

**File:** `1-td_lambtha.py`

TD(λ) combines Temporal Difference learning with eligibility traces. These traces allow the algorithm to distribute updates across previously visited states.

The parameter `lambtha` controls how quickly the influence of previous states decreases.

-   When λ = 0, the algorithm behaves like TD(0).
    
-   When λ approaches 1, earlier states retain more influence.
    

### 2\. SARSA(λ)

**File:** `2-sarsa_lambtha.py`

SARSA(λ) learns action-value estimates using state-action pairs and eligibility traces.

It uses an epsilon-greedy policy to balance exploration and exploitation.

The algorithm updates the Q-table using the reward, the next state-action value, the learning rate, and the eligibility traces.

## Key Concepts

| Concept | Description |
| --- | --- |
| State-value function V(s) | Estimates the expected return from a state. |
| Action-value function Q(s, a) | Estimates the expected return from taking an action in a state. |
| Learning rate α | Controls the size of each update. |
| Discount factor γ | Controls the importance of future rewards. |
| Eligibility traces λ | Control how much previous states or actions influence updates. |
| Exploration | Trying different actions to discover better strategies. |
| Exploitation | Choosing actions with the highest estimated value. |
| Epsilon-greedy | Selects random actions with probability ε and greedy actions otherwise. |

## Usage

Run the test scripts from the project directory:

```
./0-main.py
./1-main.py
./2-main.py
```

Each script initializes the FrozenLake environment, runs the corresponding algorithm, and displays the updated value estimates or Q-table.

## Environment

The project uses the FrozenLake 8x8 environment from Gymnasium.

The agent moves across a grid containing safe states, holes, and a goal. The objective is to learn useful state values or action values through repeated interactions with the environment.

## Author

ONIER Pauline - Holberton School