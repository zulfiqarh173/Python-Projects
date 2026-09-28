import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

""" 
Tensor operation include 

Addition
Subtraction
Multiplication (Element-wise)
Division
Matrix multiplication

"""

#Create a tensor and add 10 to it
tensor = torch.tensor([1,2,3])
print("OG Tensor")
print(tensor)
print("OG Tensor + 10")
print(tensor+10)

#Multiply tensor by 10 (This is element wise scalar multiplication)
print("OG Tensor * 10")
print(tensor*10)

#Subtract tensor by 10
print("OG Tensor - 10")
print(tensor-10)



"""
For Matrix Multiplication, there are 2 rules
1.  The inner dimensions must match:
    (3, 2) @ (3, 2) won't work
    (2, 3) @ (3, 2) will work
    (3, 2) @ (2, 3) will work

2.  The resulting matrix has the shape of the outer dimensions:
    (2, 3) @ (3, 2) -> (2, 2)
    (3, 2) @ (2, 3) -> (3, 3)

"""

#Element wise matrix multiplication
print("OG Tensor * OG Tensor")
print(tensor*tensor)

#Matrix Multiplication
print("Matrix Multiplication")
print(torch.matmul(tensor, tensor))

# This is a common Error in deep learning: shape errors

tensor_A = torch.tensor([[1,2],
                        [3,4],
                        [5,6]])

tensor_B = torch.tensor([[7,8],
                        [9,10],
                        [11,12]])

# print(torch.matmul(tensor_A, tensor_B)) -- If you run this, there will be an error

# To fix this error, use the transpose function
# Transpose switchs the axes or dimensions of a tensor
# Shape(a,b) -> Shape(b,a)


print(f"Orignal Shapes: tensorA = {tensor_A.shape}, tensorB = {tensor_B.shape}")
tensor_B = tensor_B.T
print(f"New Shapes: tensorA = {tensor_A.shape} (Same shape as before), tensorB = {tensor_B.shape}")
print(f"Multiplying: {tensor_A.shape} with {tensor_B.shape} <- inner dimensions must match")
print("Output:")
print(torch.matmul(tensor_A, tensor_B))

print(f"Output Shape: {torch.matmul(tensor_A, tensor_B).shape}")