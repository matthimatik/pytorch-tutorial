"""
Softmax example
"""

import torch
# import torch.nn as nn
import numpy as np
# pylint: disable=not-callable, missing-function-docstring


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print("Softmax numpy:", outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print("Softmax torch:", outputs)
