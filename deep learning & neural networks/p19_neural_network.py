'''
Neural Network for Fashion MNIST Classification
From AI and Machine Learning Algorithms and Techniques by Microsoft on Coursera
'''

from keras import layers, models
import keras

# Load the Fashion MNIST dataset
(train_images, train_labels), (test_images,
                               test_labels) = keras.datasets.fashion_mnist.load_data()

# Normalize the pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model
model = models.Sequential([
    # Input layer to flatten the 2D images
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons
    layers.Dense(10, activation='softmax')  # Output layer with 10 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f'Test accuracy: {test_acc}')

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    # Additional hidden layer with 64 neurons
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
