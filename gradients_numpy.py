"""
Basic machine learning with numpy
"""
import numpy as np

# f = w * x
# don't care about the bias

# f = 3 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([3, 6, 9, 12], dtype=np.float32)

w = 0.0  # this should be 3 in the end!

# model predicition


def forward(x):
    return w * x


# loss, MSE


def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# gradient
# MSE 1/N * (w * x - y)**2
# dJ/dw = 1/N 2x (w*x - y)


def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()


print(f"Prediction before training: f(5) = {forward(5):.3f}")

# Training
learning_rate = 0.01
n_iters = 20
print_amount = 10
print_rate = n_iters / print_amount

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    dw = gradient(X, Y, y_pred)

    # update weights
    w -= learning_rate * dw

    if epoch % print_rate == 0:
        print(f"Epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")
