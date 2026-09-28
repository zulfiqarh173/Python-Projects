import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

""" 
Reshaping, stacking, squeexing and unsqueezing Tensors

Reshaping - reshapes an input tensor to a defined shape
View - Return a view of an input tensor of certain shape but keep the same memory as the original tensor
Stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
Squeeze - removes all `1` dimensions from a tensor
Unsqueeze - add a `1` dimension to a target tensor
Permute - Return a view of the input with dimensions permuted (swapped) in a certain way


"""

# Lets create a tensor
x = torch.arange(1., 10.)
print(x, x.shape)

# Add an extra dimension
# Before Reshaping, consider the shape of the input tensor and your desired shape of target tensor
# The shape of both tensor has to equal
# For example, The orginal tensor has a shape (9) and our desired tensor shape (9, 1)
# This is allowed because 
# 9 = 9*1
# 9 = 9
# An example that dosent work is if the orginal tensor shape is (9) and our desired tensor shape (9,2)
# This will not work and throw an error because
# 9 != 9*2
# 9 != 18

xReshaped = x.reshape(1,9)
print(xReshaped)

# Change the View
xView = x.view(1,9)

# Changing xView will also change x (Orginal tensor) because a view of a tensor shares the same memory as the original
xView[0,0] = 5
print(xView, x)

# Stack tensors ontop each other 
xStacked = torch.stack([x,x,x,x], dim=0)
print(xStacked)

# Squeeze practice - removes all single dimensions from a target tensor
print(f"Previous tensor: {xReshaped}")
print(f"Previous shape: {xReshaped.shape}")

xReshaped = xReshaped.squeeze()

print(f"Tensor after squeeze: {xReshaped}")
print(f"Tensor shape after squeeze: {xReshaped.shape}")

# Unsqueeze practice - adds a single dimensions to a target tensor at a specific dim (dimension)
print(f"Previous tensor: {x}")
print(f"Previous shape: {x.shape}")

xUnsqueezed = x.unsqueeze(dim=0)

print(f"Tensor after unsqueeze: {xUnsqueezed}")
print(f"Tensor shape after unsqueeze: {xUnsqueezed.shape}")

#torch.permute - rearranges the dimension of a target tensor in a specified order
xOriginal = torch.rand(size=(224,224,3)) # [height, width, colour channels]

#Permute the original tensor to rearrange the axis (or dim) order
xPermuted = torch.permute(xOriginal, dims=(2, 0, 1))

print(f"Previous Shape {xOriginal.shape}")
print(f"New (Permuted) Shape {xPermuted.shape}")