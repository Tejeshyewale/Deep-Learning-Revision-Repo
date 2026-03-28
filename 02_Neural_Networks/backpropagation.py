import numpy as np

# Sample data
X = np.array([[1, 2]])
y = np.array([[1]])

# Weights
W = np.random.rand(2, 1)
b = np.zeros((1,))

# Learning rate
lr = 0.01

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Training loop
for i in range(100):
    # Forward
    z = np.dot(X, W) + b
    y_pred = sigmoid(z)

    # Loss derivative
    error = y_pred - y

    # Backprop
    dW = np.dot(X.T, error)
    db = np.sum(error)

    # Update
    W -= lr * dW
    b -= lr * db

print("Trained Weights:", W)

