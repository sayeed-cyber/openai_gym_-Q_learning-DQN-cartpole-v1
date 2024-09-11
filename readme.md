# Report: Comparing Q-Learning and Deep Q-Network (DQN) on CartPole-v1

## 1. Introduction

In this experiment, we compare two reinforcement learning (RL) algorithms: **Q-Learning** and **Deep Q-Network (DQN)**, applied to the classic **CartPole-v1** environment in OpenAI Gym. The task in CartPole-v1 is to balance a pole on a moving cart by applying forces to the cart either to the left or right. The goal of the agent is to keep the pole balanced for as long as possible.

The two algorithms differ in their approaches to solving this problem. Q-Learning is a table-based method, which requires discretizing the state space, while DQN uses a neural network to approximate the Q-values, allowing it to handle continuous state spaces. This report compares both the performance and the strengths and weaknesses of these methods based on the analysis of the provided code.

## 2. Implementation Overview

### Q-Learning Implementation

The Q-Learning implementation relies on a **Q-table** where the agent learns the Q-values for each action in every discrete state. Since CartPole has continuous states, the state space was **discretized** using bins. This reduced the precision but allowed Q-Learning to operate within a limited table size.

**Key aspects of the implementation:**
- **Epsilon-greedy strategy** for action selection, allowing exploration with probability epsilon and exploitation otherwise.
- **Q-table initialization**, where the Q-values for all state-action pairs were initialized to zero.
- **Learning rate (`alpha`)** and **discount factor (`gamma`)** were used to update the Q-values as the agent learned from its experiences.

### DQN Implementation

The DQN agent uses a **deep neural network** to approximate Q-values instead of relying on a Q-table. This allows it to work directly with the continuous state space of CartPole-v1. The neural network receives the current state as input and outputs Q-values for each possible action.

**Key aspects of the implementation:**
- **Experience replay**: The agent stores past experiences in a memory buffer and samples them randomly during training, which helps reduce correlation between consecutive experiences.
- **Target network**: A separate target network was used to stabilize training by providing a fixed target for the Q-value updates over a number of episodes.
- **Neural network architecture**: The DQN used a feedforward neural network with two hidden layers.
- **Epsilon decay**: Over time, the probability of exploring (epsilon) was reduced to encourage the agent to exploit learned policies.

## 3. Results

### Q-Learning Performance

The Q-Learning agent showed an initial improvement as it learned to balance the pole over time, but the progress was slow due to the discretization of the state space. The agent's ability to generalize was limited, resulting in inconsistent performance. Once the Q-table started to fill, learning plateaued, and it became difficult for the agent to make fine adjustments to the cart's position.

**Observations:**
- Early episodes showed significant variability in the agent's performance.
- After several hundred episodes, the agent managed to balance the pole for longer durations, though not consistently.
- The final policy learned by the Q-Learning agent could balance the pole for a moderate number of time steps but struggled with more precise control.

### DQN Performance

The DQN agent learned much faster than Q-Learning, owing to its ability to approximate Q-values for continuous states without discretization. After a relatively short training period, the DQN agent started to balance the pole for much longer periods and eventually became quite proficient.

**Observations:**
- The DQN agent showed rapid improvement after the initial exploration phase, with rewards per episode rising quickly.
- Stability in performance was observed after the agent had enough experience in the environment, and it could balance the pole consistently for long episodes.
- The use of experience replay and the target network helped smooth out learning and avoided the instability seen in Q-Learning.

## 4. Discussion: Strengths and Weaknesses

### Q-Learning

**Strengths:**
- Simple and easy to implement, especially for problems with discrete state-action spaces.
- Can perform well in environments with small, discrete state spaces.
  
**Weaknesses:**
- Struggles with continuous state spaces like CartPole, requiring discretization, which can lead to loss of precision.
- Learning was slower and less stable, especially as the state space grows larger.
- More sensitive to hyperparameters (such as learning rate, epsilon, and number of bins used for discretization).

### DQN

**Strengths:**
- Capable of handling large and continuous state spaces like CartPole-v1 without the need for discretization.
- Neural networks allow the agent to generalize better across states, resulting in faster learning and better policies.
- The use of experience replay and a target network leads to more stable learning compared to Q-Learning.

**Weaknesses:**
- More complex to implement than Q-Learning, requiring careful tuning of the neural network, learning rate, and other parameters.
- Computationally more expensive due to the neural network and the experience replay mechanism.
- Requires more memory to store experience replay, and is sensitive to the architecture of the neural network and the size of the replay buffer.

## 5. Conclusion

In this experiment, **DQN** outperformed **Q-Learning** for solving the CartPole-v1 environment. While Q-Learning struggled with the continuous nature of the environment, DQN was able to leverage its neural network to approximate Q-values directly, leading to faster and more stable learning.

For tasks like CartPole, where the state space is continuous, DQN is the preferred approach due to its flexibility and effectiveness. However, if simplicity and interpretability are more important, Q-Learning can still be useful, particularly in smaller, discrete environments.

Overall, DQN demonstrated superior performance, but its complexity and resource requirements are trade-offs to consider when choosing between the two algorithms.



[q_learning video](q_learning.mp4)