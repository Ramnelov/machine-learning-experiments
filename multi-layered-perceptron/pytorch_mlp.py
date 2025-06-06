# This code implements a simple multi-layered perceptron (MLP) usimg PyTorch.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import generate_data
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# For reproducibility
np.random.seed(123)
torch.manual_seed(123)

data = generate_data()  # shape: (num_samples, 2)

EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.01

TRAIN_SPLIT = 0.8

# Shuffle and split the data into training and testing sets
np.random.shuffle(data)
split_idx = int(TRAIN_SPLIT * data.shape[0])
train_data = data[:split_idx]
test_data = data[split_idx:]

# Convert numpy arrays to torch tensors
x_train = torch.tensor(train_data[:, :1], dtype=torch.float32)
y_train = torch.tensor(train_data[:, 1:], dtype=torch.float32)
x_test = torch.tensor(test_data[:, :1], dtype=torch.float32)
y_test = torch.tensor(test_data[:, 1:], dtype=torch.float32)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


HIDDEN_UNITS = 10

# Define the model
model = nn.Sequential(nn.Linear(1, HIDDEN_UNITS), nn.Tanh(), nn.Linear(HIDDEN_UNITS, 1))

# Train the model
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
for epoch in range(EPOCHS):

    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
        for x, y in pbar:

            optimizer.zero_grad()
            y_hat = model(x)
            loss = F.mse_loss(y_hat, y)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        loss = F.mse_loss(y_pred, y_test)

    model.train()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss:.4f}")

# Final evaluation on the test set
model.eval()
with torch.no_grad():
    y_pred = model(x_test)
    loss = F.mse_loss(y_pred, y_test)

print(f"Final Loss: {loss:.4f}")

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(x_test, y_test, label="Ground truth", color="blue", s=5)
plt.scatter(x_test, y_pred, label="Predictions", color="red", s=5, alpha=0.5)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Final Predictions vs Ground Truth")
plt.legend()
plt.show()
