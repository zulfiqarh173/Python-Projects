import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

''' 
Why random Tensors?

Random Tensors are important because many neural networks learn is that they start with tensors
full of random valuess and then adjust those random numbers to better represent the data.

Start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers.....
'''

randomTensor = torch.rand(3,4)

# Create random tensor of size (3,4)
randomTensorImage = torch.rand(size=(224,224,3))
print(randomTensorImage[10][0])

# Create a tensor of zeros
zeros = torch.zeros(size=(3,4))

#create a tensor of ones
ones = torch.ones(size=(3,4,3))
print(ones)
print(ones.dtype)