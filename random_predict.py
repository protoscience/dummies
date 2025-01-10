#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Generate some example data
# Create a random input tensor of size (100, 1)
X = torch.randn(100, 1)  # 100 random data points
y = X * 2 + 1  # Linear relationship: y = 2 * X + 1

# Step 2: Create a simple feed-forward neural network (1 hidden layer)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Input layer to hidden layer (1 -> 10)
        self.fc2 = nn.Linear(10, 1)  # Hidden layer to output layer (10 -> 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation after first layer
        x = self.fc2(x)          # Output layer
        return x

# Step 3: Create the model instance
model = SimpleNN()

# Step 4: Define the loss function and the optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss (for regression)
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Step 5: Create DataLoader to handle mini-batches
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)  # Batch size of 10

# Step 6: Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    for data, target in dataloader:
        optimizer.zero_grad()  # Zero the gradients from previous step
        output = model(data)   # Forward pass through the model
        loss = criterion(output, target)  # Compute loss
        loss.backward()        # Backpropagate to compute gradients
        optimizer.step()       # Update weights using gradients

    if epoch % 10 == 0:  # Print loss every 10 epochs
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 7: Test the model after training
with torch.no_grad():  # Turn off gradient tracking (we don't need it for testing)
    test_input = torch.tensor([[5.0]])  # Test with a sample input
    prediction = model(test_input)  # Get the model's prediction
    print(f'Prediction for input 5: {prediction.item():.4f}')  # Print the predicted value