import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Tensor datatypes is one of the 3 errors you'll run into with pytorch and deep learning
# Tensors not right datatype
# Tensors not right shape
# Tensors not right device

float_32_tensor = torch.tensor([3.0, 6.0, 9.0], 
                               dtype=None, # what datatype the tesnor is
                               device="cuda", # what device is the tensor on
                               requires_grad=False) # whether or not to track gradients with this tensor's operation


float_16_tensor = float_32_tensor.type(torch.float16)



# Tensors not right datatype - To get datatype, use tensor.datatype
# Tensors not right shape - To get shape, use tensor.shape
# Tensors not right device - To get device , use tensor.device

someTensor = torch.rand(size=(2,4,6),
                        dtype=torch.float16,
                        device="cuda")

print(someTensor)
print(f"Datatype of datatype: {someTensor.dtype}\nShape of Tensor: {someTensor.shape}\nDevice of Tensor: {someTensor.device}")