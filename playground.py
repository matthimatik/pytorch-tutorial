"""
My playground file for pytorch
"""

import torch

x = torch.FloatTensor(2,2)
y = torch.rand(2,2)
print(x)
print(y)
z = x + y
print(z.shape)
