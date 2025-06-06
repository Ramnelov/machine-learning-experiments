# This code implements a simple multi-layered perceptron (MLP) from scratch using NumPy.

import matplotlib.pyplot as plt
import numpy as np
from data import generate_data

np.random.seed(123)  # For reproducibility

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

HIDDEN_UNITS = 10

# Initialize weights and biases in the range [-1, 1]
W1 = np.random.uniform(-1, 1, size=(1, HIDDEN_UNITS))
b1 = np.random.uniform(-1, 1, size=(HIDDEN_UNITS,))
W2 = np.random.uniform(-1, 1, size=(HIDDEN_UNITS, 1))
b2 = np.random.uniform(-1, 1, size=(1,))


def forward_pass(x: np.ndarray) -> np.ndarray:
    """Perform a forward pass through the network."""

    z = np.dot(x, W1) + b1
    h = np.tanh(z)
    y_hat = np.dot(h, W2) + b2

    return y_hat


def backpropagation(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute gradients for weights and biases using backpropagation."""

    z = np.dot(x, W1) + b1
    h = np.tanh(z)
    y_hat = np.dot(h, W2) + b2

    error = y_hat - y
    grad_y_hat = error / x.shape[0]

    grad_W2 = np.dot(h.T, grad_y_hat)
    grad_b2 = np.sum(grad_y_hat, axis=0)

    grad_h = np.dot(grad_y_hat, W2.T)
    grad_z = grad_h * (1 - h**2)

    grad_W1 = np.dot(x.T, grad_z)
    grad_b1 = np.sum(grad_z, axis=0)

    return grad_W1, grad_b1, grad_W2, grad_b2


# Train the model
for epoch in range(EPOCHS):

    np.random.shuffle(train_data)
    num_samples = train_data.shape[0]

    for i in range(0, num_samples, BATCH_SIZE):
        batch = train_data[i : i + BATCH_SIZE]

        # shape: (batch_size, 1)
        x = batch[:, :1]
        y = batch[:, 1:]

        grad_W1, grad_b1, grad_W2, grad_b2 = backpropagation(x, y)

        # Update weights and biases using gradients
        W1 -= LEARNING_RATE * grad_W1
        b1 -= LEARNING_RATE * grad_b1
        W2 -= LEARNING_RATE * grad_W2
        b2 -= LEARNING_RATE * grad_b2

    loss = 0
    for i in range(0, test_data.shape[0], BATCH_SIZE):
        batch = test_data[i : i + BATCH_SIZE]

        # shape: (batch_size, 1)
        x = batch[:, :1]
        y = batch[:, 1:]

        y_hat = forward_pass(x)

        loss += np.sum((y_hat - y) ** 2)

    loss /= test_data.shape[0]

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss:.4f}")

# Final evaluation on the test set
x_test = test_data[:, :1]
y_test = test_data[:, 1:]
y_pred = forward_pass(x_test)
y_true = np.sin(x_test)

loss = np.mean((y_pred - y_test) ** 2)
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
