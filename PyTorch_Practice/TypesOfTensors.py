import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Creating Tensors

# Scalar
print("SCALAR STUFF")
scalar = torch.tensor(7)
print(scalar.ndim)

# Get tensor back as python int
print(scalar.item())

# Vector
print("VECTOR STUFF")
vector = torch.tensor([7,7])
print(vector)
print(vector.ndim)
print(vector.shape)
print(vector[0].item())

# MATRIX
print("MATRIX STUFF")
MATRIX = torch.tensor([[7,7],
                       [6,6]])

print(MATRIX)
print(MATRIX.ndim)
print(MATRIX.shape)
print(MATRIX[0])
print(MATRIX[1][0].item())

# TENSOR (Big Boy Stuff)
print("TENSOR STUFF")
TENSOR = torch.tensor([[[1,2,3], [4,5,6]],
                       [[7,8,9], [10,20,30]]])

print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape)

