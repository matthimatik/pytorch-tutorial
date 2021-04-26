"""
Dataset and DataLoader - Batch Training
"""
import math

import numpy as np
import torch
# import torchvision
from torch.utils.data import DataLoader, Dataset


class WineDataset(Dataset):

    def __init__(self) -> None:
        # data loading
        # pylint: disable=invalid-name
        xy = np.loadtxt('./data/wine.csv', delimiter=",",
                        dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        # The "[0]" converts to column vector
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples


my_dataset = WineDataset()
# print(len(my_dataset))
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)

BATCH_SIZE = 4
my_dataloader = DataLoader(
    dataset=my_dataset, batch_size=BATCH_SIZE, shuffle=True)  # num_workers>0 currently not working

# dataiter = iter(my_dataloader)
# data = dataiter.next()
# features, labels = data
# print(features, labels)

# training loop
NUM_EPOCHS = 2

total_samples = len(my_dataset)
n_iterations = math.ceil(total_samples / BATCH_SIZE)
print(total_samples, n_iterations)


for epoch in range(NUM_EPOCHS):
    for i, (inputs, labels) in enumerate(my_dataloader):
        # forward, backward, update
        if (i+1) % 5 == 0:
            print(
                f"epoch {epoch+1}/{NUM_EPOCHS},\t step {i+1}/{n_iterations},\t inputs {inputs.shape}")


# useful example datasets
# torchvision.datasets.MNIST, fashion-mnist, cifar, coco
