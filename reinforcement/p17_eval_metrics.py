'''
Reinforcement Learning Evaluation Metrics Example
From AI and Machine Learning Algorithms and Techniques by Microsoft on Coursera
'''

import matplotlib.pyplot as plt
import numpy as np

# Define the grid size and number of states
GRID_SIZE = 5
N_STATES = GRID_SIZE * GRID_SIZE

# Define the reward structure: -1 for all states, +10 for the goal, -10 for the pitfall
rewards = np.full((N_STATES,), -1)  # Default reward of -1
rewards[24] = 10  # Goal state at position 24 (bottom-right)
rewards[12] = -10  # Pitfall at position 12 (center)

# Define the number of actions (up, down, left, right)
N_ACTIONS = 4


def epsilon_greedy_action(Q_table, state, epsilon):
    """Select an action using the epsilon-greedy strategy."""
    # Epsilon-greedy strategy: with probability epsilon, take a random action (exploration)
    # otherwise take the action with the highest Q-value for the given state (exploitation)
    if np.random.rand() < epsilon:  # Exploration
        return np.random.randint(0, Q_table.shape[1])  # Random action
    else:  # Exploitation
        return np.argmax(Q_table[state])  # Action with the highest Q-value


ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.1  # Exploration rate for epsilon-greedy policy

# Initialize the Q-table
Q_table = np.zeros((N_STATES, N_ACTIONS))

# Training loop
for episode in range(1000):
    state = np.random.randint(0, N_STATES)  # Start at random state
    DONE = False
    while not DONE:
        action = epsilon_greedy_action(Q_table, state, EPSILON)
        next_state = np.random.randint(0, N_STATES)  # Random next state
        reward = rewards[next_state]

        # Q-learning update rule
        Q_table[state, action] = Q_table[state, action] + ALPHA * \
            (reward + GAMMA *
             np.max(Q_table[next_state]) - Q_table[state, action])

        state = next_state
        if next_state == 24 or next_state == 12:  # End episode if goal or pitfall is reached
            DONE = True

# Redefine epsilon_greedy_action to log explorations & exploitations
actions = []

def epsilon_greedy_action_with_logging(Q_table, state, epsilon):
    """Select an action using the epsilon-greedy strategy with action logging."""
    # Epsilon-greedy strategy: with probability epsilon, take a random action (exploration)
    # otherwise take the action with the highest Q-value for the given state (exploitation)
    if np.random.rand() < epsilon:  # Exploration
        actions.append('explore')
        return np.random.randint(0, Q_table.shape[1])  # Random action
    else:  # Exploitation
        actions.append('exploit')
        return np.argmax(Q_table[state])  # Action with the highest Q-value


# Calculate and store cumulative rewards
cumulative_rewards = []
for episode in range(1000):
    TOTAL_REWARD = 0
    state = np.random.randint(0, N_STATES)
    DONE = False
    while not DONE:
        action = epsilon_greedy_action_with_logging(Q_table, state, EPSILON)
        next_state = np.random.randint(0, N_STATES)
        reward = rewards[next_state]
        TOTAL_REWARD += reward
        state = next_state
        if next_state == 24 or next_state == 12:
            DONE = True
    cumulative_rewards.append(TOTAL_REWARD)

# Plot the cumulative rewards over episodes
plt.plot(cumulative_rewards)
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward Over Episodes')
plt.show()

# Calculate and store episode lengths
episode_lengths = []
actions = []
for episode in range(1000):
    STEPS = 0
    state = np.random.randint(0, N_STATES)
    DONE = False
    while not DONE:
        action = epsilon_greedy_action(Q_table, state, EPSILON)

        next_state = np.random.randint(0, N_STATES)

# Calculate and store cumulative rewards and actions
cumulative_rewards = []
for episode in range(1000):
    TOTAL_REWARD = 0
    state = np.random.randint(0, N_STATES)
    DONE = False
    while not DONE:
        action = epsilon_greedy_action(Q_table, state, EPSILON)
        next_state = np.random.randint(0, N_STATES)
        reward = rewards[next_state]
        TOTAL_REWARD += reward
        state = next_state
        if next_state == 24 or next_state == 12:
            DONE = True
    cumulative_rewards.append(TOTAL_REWARD)
for episode in range(1000):
    TOTAL_REWARD = 0
    state = np.random.randint(0, N_STATES)
    DONE = False
    while not DONE:
        action = epsilon_greedy_action(Q_table, state, EPSILON)
        next_state = np.random.randint(0, N_STATES)
        reward = rewards[next_state]
        TOTAL_REWARD += reward
        state = next_state
        if next_state == 24 or next_state == 12:
            DONE = True
    cumulative_rewards.append(TOTAL_REWARD)

# Calculate success rate
success_count = sum(1 for reward in cumulative_rewards if reward >= 10)
success_rate = success_count / len(cumulative_rewards)

# Exploration vs. exploitation ratio
# print(actions)
exploration_count = sum(1 for action in actions if action == 'explore')
exploitation_count = sum(1 for action in actions if action == 'exploit')
exploration_exploitation_ratio = exploration_count / \
    (exploration_count + exploitation_count)

print(f"Success Rate: {success_rate * 100}%")
print(f"Exploration vs. Exploitation Ratio: {exploration_exploitation_ratio}")
