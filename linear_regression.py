"""
Linear regression with pytorch
"""
# 1) Design Model (input size, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) prepare data
X_numpy, y_numpy = datasets.make_regression(  # pylint: disable=unbalanced-tuple-unpacking
    n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape


# 1) model
input_size = n_features
OUTPUT_SIZE = 1

model = nn.Linear(input_size, OUTPUT_SIZE)


# 2) loss and optimizer
LEARNING_RATE = 0.1
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


# 3) training loop
NUM_EPOCHS = 100
PRINT_AMOUNT = 10

for epoch in range(NUM_EPOCHS):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # backward pass
    loss.backward()

    # update
    optimizer.step()
    optimizer.zero_grad()

    if epoch % (NUM_EPOCHS / PRINT_AMOUNT) == 0 or epoch == NUM_EPOCHS - 1:
        [w, b] = model.parameters()
        print(
            f"Epoch {epoch}:\t w = {w[0][0].item():.3f}, loss = {loss.item():.8f}")

# plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
