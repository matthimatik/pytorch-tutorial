"""
Logistic regression with pytorch
"""
# 1) Design Model (input size, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights

import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 0) prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target  # pylint: disable=no-member

n_samples, n_features = X.shape
print(f"n_samples: {n_samples}, n_features: {n_features}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1000)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# no fit necessary, because mean and std are the same like X_train
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

print(y_train.shape)
y_train = y_train.view(y_train.shape[0], 1)
print(y_train.shape)
y_test = y_test.view(y_test.shape[0], 1)


# 1) model
# f = sig(wx + b)


class LogsticRegression(nn.Module):
    """Logistic Regression Module"""

    def __init__(self, n_input_features):
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, data):
        """Forward pass through the model"""
        return torch.sigmoid(self.linear(data))


model = LogsticRegression(n_features)


# 2) loss optimizer
LEARNING_RATE = 0.05
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


# 3) training loop
NUM_EPOCHS = 1000
PRINT_AMOUNT = 10

for epoch in range(NUM_EPOCHS):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()

    # updates
    optimizer.step()
    optimizer.zero_grad()

    if epoch % (NUM_EPOCHS / PRINT_AMOUNT) == 0 or epoch == NUM_EPOCHS - 1:
        [w, b] = model.parameters()
        print(
            f"Epoch {epoch}:\t w = {w[0][0].item():.3f},\t loss = {loss.item():.8f}")


# 4) evalutation
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"accuracy = {acc:.4f}")
