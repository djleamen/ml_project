'''
Deep Learning and Generative AI with Autoencoders and GANs
From AI and Machine Learning Algorithms and Techniques by Microsoft on Coursera
'''

from keras import layers, models, datasets
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, _), (X_test, _) = datasets.mnist.load_data()

# Normalize and flatten images
X_train = np.array(X_train).astype('float32') / 255.
X_test = np.array(X_test).astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

# Define the encoder
encoder = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu')  # Bottleneck layer
])

# Define the decoder
decoder = models.Sequential([
    layers.Input(shape=(64,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(784, activation='sigmoid')  # Reconstructed output
])

# Build the autoencoder model
autoencoder = models.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=20, batch_size=256, validation_data=(X_test, X_test))

# Predict reconstructed images
reconstructed_images = autoencoder.predict(X_test)

# Calculate the mean squared error
mse = np.mean(np.square(X_test - reconstructed_images))
print(f'Autoencoder Reconstruction MSE: {mse}')

# Define the generator
def build_generator():
    """Builds the generator model for GAN."""
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(100,)),
        layers.Dense(784, activation='sigmoid')  # Output: 28x28 flattened image
    ])
    return model

# Define the discriminator
def build_discriminator():
    """Builds the discriminator model for GAN."""
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(1, activation='sigmoid')  # Output: Probability (real or fake)
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Stack the generator and discriminator
gan = models.Sequential([generator, discriminator])
discriminator.trainable = False
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training GAN
EPOCHS = 10000
BATCH_SIZE = 64
HALF_BATCH = BATCH_SIZE // 2

for epoch in range(EPOCHS):
    # Real images
    idx = np.random.randint(0, X_train.shape[0], HALF_BATCH)
    real_images = X_train[idx]
    real_labels = np.ones((HALF_BATCH, 1))

    # Fake images
    noise = np.random.normal(0, 1, (HALF_BATCH, 100))
    fake_images = generator.predict(noise)
    fake_labels = np.zeros((HALF_BATCH, 1))

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

    # Train the generator (via GAN model)
    noise = np.random.normal(0, 1, (BATCH_SIZE, 100))
    gan_labels = np.ones((BATCH_SIZE, 1))  # Try to fool the discriminator
    g_loss = gan.train_on_batch(noise, gan_labels)

    # Every 1000 epochs, print losses and visualize generated images
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Discriminator Loss: {d_loss_real[0]}, Generator Loss: {g_loss}")
        # Generate and display images
        generated_images = generator.predict(np.random.normal(0, 1, (10, 100)))
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
        plt.show()
