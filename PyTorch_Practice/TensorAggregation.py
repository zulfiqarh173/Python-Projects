import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create a Tensor
x = torch.arange(0, 100, 10)


#The mean function will not work without this line below
x = x.type(torch.float)

print("Tensor is .....")
print(x)

# Find the min
print(f"Tensor Minimum is {torch.min(x)}")

# Find the max
print(f"Tensor Maximum is {torch.max(x)}")

# Find the max
print(f"Tensor average is {torch.mean(x)}")

# Find the sum
print(f"Tensor sum is {torch.sum(x)}")

#Find the index of max value
print(f"The position of the max value is at: {int(torch.argmax(x).item())}")

#Find the index of min value
print(f"The position of the min value is at: {int(torch.argmin(x).item())}")