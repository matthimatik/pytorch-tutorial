"""
Basic machine learning with pytorch
"""

# 1) Design Model (input size, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights

import torch
import torch.nn as nn
# pylint: disable=not-callable

# Training data
# f = 3 * x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[3], [6], [9], [12]], dtype=torch.float32)


n_samples, n_features = X.shape
# print(n_samples, n_features)

input_size = n_features
output_size = n_features


class LinearRegression(nn.Module):
    """My wrapper class for linear regression model"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, data):
        """Forward pass through the model"""
        return self.lin(data)


# model = nn.Linear(input_size, output_size)
model = LinearRegression(input_size, output_size)  # We use our own model

X_test = torch.tensor([5], dtype=torch.float32)

print(f"Prediction before training: f(5) = {model(X_test).item():.3f}")


# Training
LEARNING_RATE = 0.1
N_ITERS = 300
PRINT_AMOUNT = 10

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters, lr=LEARNING_RATE)

for epoch in range(N_ITERS):
    # prediction = forward pass
    y_pred = model(X)

    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward()  # dl/dw

    # update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % (N_ITERS / PRINT_AMOUNT) == 0:
        [w, b] = model.parameters()
        print(f"Epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")

print(f"Prediction after training: f(5) = {model(X_test).item():.3f}")
