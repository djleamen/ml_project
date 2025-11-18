'''
Implementation of Feedforward Neural Networks (FNNs), 
Convolutional Neural Networks (CNNs), and Recurrent Neural Networks (RNNs)
From AI and Machine Learning Algorithms and Techniques by Microsoft on Coursera
'''

import numpy as np
from keras import layers, models
from keras.datasets import cifar10

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load dataset
X, y_raw = load_iris(return_X_y=True)
y = np.array(y_raw).reshape(-1, 1)  # pylint: disable=no-member

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Build the FNN model
model_fnn = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    # 3 output classes for the Iris dataset
    layers.Dense(3, activation='softmax')
])

# Compile the model
model_fnn.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_fnn.fit(X_train, y_train, epochs=20, batch_size=32,
              validation_data=(X_test, y_test))

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Build the CNN model
model_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 output classes for CIFAR-10
])

# Compile the model
model_cnn.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_cnn.fit(train_images, train_labels, epochs=10, batch_size=64,
              validation_data=(test_images, test_labels))


# Generate synthetic sine wave data
t = np.linspace(0, 100, 10000)
X = np.sin(t).reshape(-1, 1)

# Prepare sequences
def create_sequences(data, seq_length):
    """Create sequences of data for RNN input."""
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(data[i+seq_length])
    return np.array(X_seq), np.array(y_seq)


SEQ_LENGTH = 100
X_seq, y_seq = create_sequences(X, SEQ_LENGTH)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42)

# Build the RNN model
model_rnn = models.Sequential([
    layers.SimpleRNN(128, input_shape=(SEQ_LENGTH, 1)),
    layers.Dense(1)  # Output is a single value (next point in the sequence)
])

# Compile the model
model_rnn.compile(optimizer='adam', loss='mse')

# Train the model
model_rnn.fit(X_train, y_train, epochs=10, batch_size=32,
              validation_data=(X_test, y_test))
