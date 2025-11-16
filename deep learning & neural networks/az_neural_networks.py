from azureml.core import Workspace

# Create or retrieve an existing Azure ML workspace
ws = Workspace.get(name='myworkspace',
                      subscription_id='your-subscription-id',
                      resource_group='myresourcegroup',
                      location='eastus')

ws = Workspace.create(name='myworkspace',
                      subscription_id='your-subscription-id',
                      resource_group='myresourcegroup',
                      location='eastus')

# Write configuration to the workspace config file
ws.write_config(path='.azureml')

import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network with one hidden layer
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input layer (784 input features)
        self.fc2 = nn.Linear(128, 10)   # Output layer (10 classes)
        self.relu = nn.ReLU()           # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply ReLU after the first layer
        x = self.fc2(x)             # Output layer
        return x

# Initialize the neural network
model = SimpleNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For classification tasks
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

from torchvision import datasets, transforms

# Load MNIST dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=32, shuffle=True)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Reset gradients

        # Forward pass
        outputs = model(inputs.view(-1, 784))  # Flatten input
        loss = criterion(outputs, labels)      # Compute loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    
from azureml.core import Model

# Save the trained model
torch.save(model.state_dict(), 'simple_nn.pth')

# Register the model in Azure
model = Model.register(workspace=ws, model_path='simple_nn.pth', model_name='simple_nn')

# Deploying as a web service requires more steps such as creating a scoring script and environment.
