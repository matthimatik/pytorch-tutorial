"""
Feedforward NN with all techniques learned before
"""

# MNIST
# DataLoader, Transformation
# Multilayer Neural net, activation function
# Loss and optimizer
# Training loop (batch training)
# Model evaluation
# GPU support

# from time import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# hyper parameters
INPUT_SIZE = 784  # 28x28
HIDDEN_SIZE = 500
NUM_CLASSES = 10
NUM_EPOCHS = 2
BATCH_SIZE = 100
LEARNING_RATE = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

examples = iter(test_loader)
example_data, example_targets = examples.next()
print(example_data.shape, example_targets.shape)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(example_data[i][0], cmap='gray')
# plt.show()


class NeuralNet(nn.Module):
    """Neural net for digit classification"""

    def __init__(self, input_size, hidden_size, num_classes):
        # pylint: disable=invalid-name
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """Forward run through the network"""
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation nor softmax at the end
        return out


# tic = time()  # training time measurement

model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)


# loss and optimizer
loss_fnc = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# training loop
n_total_steps = len(train_loader)  # amount of batches

for epoch in range(NUM_EPOCHS):
    for i, (images, example_targets) in enumerate(train_loader):
        # 100, 1, 28, 28
        # 100, 784
        images = images.reshape(-1, 28*28).to(device)
        example_targets = example_targets.to(device)

        # forward
        outputs = model(images)
        loss = loss_fnc(outputs, example_targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(
                f"epoch {epoch+1} / {NUM_EPOCHS}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}")

# time measurement
# toc = time()
# print(f"Training time: {toc-tic}")

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # value, index; predictions are class-labels
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')
