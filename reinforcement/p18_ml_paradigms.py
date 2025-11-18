'''
Examples of different machine learning paradigms:
1. Supervised learning: Predicting house prices
2. Unsupervised learning: Customer segmentation
3. Reinforcement learning: Training an AI agent to play tic-tac-toe
From AI and Machine Learning Algorithms and Techniques by Microsoft on Coursera
'''

# ----------------------------------------------------------------------------
# Supervised learning: Predicting house prices

import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X = np.array([[2000, 3, 1], [1500, 2, 2], [
             1800, 3, 3], [1200, 2, 1], [2200, 4, 2]])
y = np.array([500000, 350000, 450000, 300000, 550000])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# ----------------------------------------------------------------------------
# Unsupervised learning: Customer segmentation

# Sample customer data: number of purchases, total spending, product categories purchased
X = np.array([[5, 1000, 2], [10, 5000, 5], [
             2, 500, 1], [8, 3000, 3], [12, 6000, 6]])

# Create and fit the KMeans model
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Print the cluster centers and labels
print(f"Cluster Centers: {kmeans.cluster_centers_}")
print(f"Labels: {kmeans.labels_}")

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Number of Purchases')
plt.ylabel('Total Spending')
plt.title('Customer Segmentation using K-Means Clustering')

# ----------------------------------------------------------------------------
# Reinforcement learning: Training an AI agent to play tic-tac-toe

# Initialize the Q-table
Q = {}

# Define the Tic-Tac-Toe board


def initialize_board():
    """Initialize an empty Tic-Tac-Toe board."""
    return np.zeros((3, 3), dtype=int)

# Check for a win


def check_win(board, player):
    """Check if the given player has won."""
    for i in range(3):
        if np.all(board[i, :] == player) or np.all(board[:, i] == player):
            return True
    if board[0, 0] == board[1, 1] == board[2, 2] == player or board[0, 2] == board[1, 1] == board[2, 0] == player:
        return True
    return False

# Check for a draw


def check_draw(board):
    """Check if the game is a draw."""
    return not np.any(board == 0)

# Get available actions


def get_available_actions(board):
    """Return a list of available actions (empty cells) on the board."""
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]

# Choose an action using epsilon-greedy policy


def choose_action(state, board, epsilon):
    """Choose an action based on epsilon-greedy policy."""
    if random.uniform(0, 1) < epsilon:
        return random.choice(get_available_actions(board))
    else:
        if state in Q and Q[state]:
            # Choose the action with the maximum Q-value
            return max(Q[state], key=Q[state].get)
        else:
            # No action in Q-table, choose random
            return random.choice(get_available_actions(board))

# Update Q-value


def update_q_value(state, action, reward, next_state, alpha, gamma):
    """Update the Q-value for the given state and action."""
    max_future_q = max(Q.get(next_state, {}).values(), default=0)
    current_q = Q.get(state, {}).get(action, 0)
    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
    if state not in Q:
        Q[state] = {}
    Q[state][action] = new_q

# Convert board to a tuple (hashable type)


def board_to_tuple(board):
    """Convert the board to a tuple for use as a key in the Q-table."""
    return tuple(map(tuple, board))

# Train the agent


def train(episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    """Train the tic-tac-toe agent using Q-learning."""
    win_history = []
    for episode in range(episodes):
        board = initialize_board()
        state = board_to_tuple(board)
        done = False
        result = None  # Initialize result
        while not done:
            action = choose_action(state, board, epsilon)
            board[action[0], action[1]] = 1
            next_state = board_to_tuple(board)
            if check_win(board, 1):
                update_q_value(state, action, 1, next_state, alpha, gamma)
                result = 1  # Agent won
                done = True
            elif check_draw(board):
                update_q_value(state, action, 0.5, next_state, alpha, gamma)
                result = 0  # Draw
                done = True
            else:
                opponent_action = random.choice(get_available_actions(board))
                board[opponent_action[0], opponent_action[1]] = -1
                next_state = board_to_tuple(board)
                if check_win(board, -1):
                    update_q_value(state, action, -1, next_state, alpha, gamma)
                    result = -1  # Agent lost
                    done = True
                elif check_draw(board):
                    update_q_value(state, action, 0.5,
                                   next_state, alpha, gamma)
                    result = 0  # Draw
                    done = True
                else:
                    update_q_value(state, action, 0, next_state, alpha, gamma)
            state = next_state
        # Record the result
        if result == 1:
            win_history.append(1)
        else:
            win_history.append(0)
    return win_history


# Train the agent for 10000 episodes
win_history = train(10000)

# Calculate the moving average of win rate


def moving_average(data, window_size):
    """Calculate the moving average of the data with the given window size."""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


# Set the window size for the moving average
WINDOW_SIZE = 100

# Compute the moving average
win_rate = moving_average(win_history, WINDOW_SIZE)

# Generate episodes for plotting
episodes = np.arange(WINDOW_SIZE, len(win_history) + 1)

# Plot the win rate over time
plt.figure(figsize=(12, 6))
plt.plot(episodes, win_rate, label='Win Rate')
plt.xlabel('Episodes')
plt.ylabel('Win Rate')
plt.title(
    f'Agent Win Rate Over Time (Moving Average over {WINDOW_SIZE} episodes)')
plt.legend()
plt.show()
