'''
Q-Learning and Policy Gradient Implementation in a Grid Environment
From AI and Machine Learning Algorithms and Techniques by Microsoft on Coursera
'''

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

# Define the grid size and actions
GRID_SIZE = 5
N_ACTIONS = 4  # Actions: up, down, left, right

# Initialize the Q-table with zeros
Q_table = np.zeros((GRID_SIZE * GRID_SIZE, N_ACTIONS))
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor for future rewards
EPSILON = 0.1  # Exploration rate for epsilon-greedy policy

# Reward matrix for the grid environment
rewards = np.full((GRID_SIZE * GRID_SIZE,), -1)  # -1 for every state
rewards[24] = 10  # Goal state
rewards[12] = -10  # Pitfall state

def epsilon_greedy_action(Q_table, state, epsilon):
    """Epsilon-greedy action selection."""
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, N_ACTIONS)  # Explore: random action
    else:
        # Exploit: action with highest Q-value
        return np.argmax(Q_table[state])


# Track rewards for Q-learning
rewards_q_learning = []

for episode in range(1000):
    # Start in a random state
    state = np.random.randint(0, GRID_SIZE * GRID_SIZE)
    DONE = False
    EPISODE_REWARD = 0
    while not DONE:
        action = epsilon_greedy_action(Q_table, state, EPSILON)
        next_state = np.random.randint(
            0, GRID_SIZE * GRID_SIZE)  # Simulated next state
        reward = rewards[next_state]
        EPISODE_REWARD += reward

        # Update Q-value using Bellman equation
        Q_table[state, action] = Q_table[state, action] + ALPHA * \
            (reward + GAMMA *
             np.max(Q_table[next_state]) - Q_table[state, action])

        state = next_state
        if next_state == 24 or next_state == 12:
            DONE = True

    rewards_q_learning.append(EPISODE_REWARD)

# Define the policy network
N_STATES = GRID_SIZE * GRID_SIZE  # Number of states in the grid

model = keras.Sequential([
    keras.layers.Dense(24, activation='relu', input_shape=(N_STATES,)),
    # Output action probabilities
    keras.layers.Dense(N_ACTIONS, activation='softmax')
])

# Optimizer for policy network updates
optimizer = keras.optimizers.Adam(learning_rate=0.01)

def get_action(state):
    """Select action using the policy network."""
    state_input = tf.one_hot([state], N_STATES)
    action_probs = model(state_input)
    action = tf.random.categorical(tf.math.log(action_probs), 1)[0, 0]
    return int(action.numpy())


# Simulation loop
states = []
actions = []
episode_rewards = []
rewards_policy_gradients = []

for episode in range(1000):
    state = np.random.randint(0, N_STATES)  # Start in a random state
    DONE = False
    EPISODE_REWARD = 0
    while not DONE:
        action = get_action(state)  # Use the provided function
        next_state = np.random.randint(0, N_STATES)  # Simulated next state
        reward = rewards[next_state]
        EPISODE_REWARD += reward

        # Store the state-action-reward trajectory
        states.append(state)
        actions.append(action)
        episode_rewards.append(reward)

        state = next_state
        if next_state in {24, 12}:
            DONE = True

    rewards_policy_gradients.append(EPISODE_REWARD)

def compute_cumulative_rewards(rewards, gamma=0.99):
    """Compute cumulative rewards for policy gradient."""
    cumulative_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        cumulative_rewards[t] = running_add
    return cumulative_rewards

def update_policy(states, actions, rewards):
    """Update the policy network using policy gradient."""
    cumulative_rewards = compute_cumulative_rewards(rewards)

    with tf.GradientTape() as tape:
        # Convert states to one-hot encoding
        state_inputs = tf.one_hot(states, N_STATES)
        action_probs = model(state_inputs)
        # Mask for selected actions
        action_masks = tf.one_hot(actions, N_ACTIONS)
        log_probs = tf.reduce_sum(
            action_masks * tf.math.log(action_probs), axis=1)

        # Policy loss is the negative log-probability of the action times the cumulative reward
        loss = -tf.reduce_mean(log_probs * cumulative_rewards)

    # Apply gradients to update the policy network
    grads = tape.gradient(loss, model.trainable_variables)
    if grads is not None:
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


# Example code to visualize rewards over episodes
plt.plot(rewards_q_learning, label='Q-Learning')
plt.plot(rewards_policy_gradients, label='Policy Gradients')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Rewards')
plt.legend()
plt.show()
