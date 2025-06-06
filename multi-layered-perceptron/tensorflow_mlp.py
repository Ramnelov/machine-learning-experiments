# This code implements a simple multi-layered perceptron (MLP) using TensorFlow.

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data import generate_data

# For reproducibility
np.random.seed(123)
tf.random.set_seed(123)

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

# Split features and targets
x_train = train_data[:, :1]
y_train = train_data[:, 1:]
x_test = test_data[:, :1]
y_test = test_data[:, 1:]

HIDDEN_UNITS = 10

# Define the model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(HIDDEN_UNITS, activation="tanh", input_shape=(1,)),
        tf.keras.layers.Dense(1),
    ]
)

# Train the model
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE), loss="mse"
)

history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test),
    verbose=1,
)

# Final evaluation on the test set
loss = model.evaluate(x_test, y_test, verbose=0)
print(f"Final Loss: {loss:.4f}")

y_pred = model.predict(x_test)

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(x_test, y_test, label="Ground truth", color="blue", s=5)
plt.scatter(x_test, y_pred, label="Predictions", color="red", s=5, alpha=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Final Predictions vs Ground Truth")
plt.legend()
plt.show()
