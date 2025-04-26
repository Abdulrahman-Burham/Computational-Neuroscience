import numpy as np

# Define RNN architecture
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize parameters
np.random.seed(0)
Wx = np.random.randn(3, 3)  # Input-to-hidden weights
Wh = np.random.randn(3, 3)  # Hidden-to-hidden weights
Wy = np.random.randn(3, 1)  # Hidden-to-output weights
learning_rate = 0.01

# Dummy data
X = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0]])
Y = np.array([[1], [0], [1]])

# Forward and backward propagation
for epoch in range(1000):
    h_prev = np.zeros((1, 3))
    loss = 0
    dWx, dWh, dWy = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(Wy)

    for t in range(X.shape[0]):
        # Forward pass
        x_t = X[t].reshape(1, -1)
        h_t = sigmoid(np.dot(x_t, Wx) + np.dot(h_prev, Wh))
        y_pred = sigmoid(np.dot(h_t, Wy))
        
        # Calculate loss
        loss += 0.5 * (Y[t] - y_pred)**2
        
        # Backward pass
        dy = -(Y[t] - y_pred) * sigmoid_derivative(y_pred)
        dWy += np.dot(h_t.T, dy)
        
        dh = np.dot(dy, Wy.T) * sigmoid_derivative(h_t)
        dWx += np.dot(x_t.T, dh)
        dWh += np.dot(h_prev.T, dh)
        
        h_prev = h_t
    
    # Update weights
    Wx -= learning_rate * dWx
    Wh -= learning_rate * dWh
    Wy -= learning_rate * dWy

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.flatten()[0]}")

print("Training complete!")
