import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# create ranged tensor
oneToTen = torch.arange(start=1, end=11, step=1)
print(oneToTen)

# create tensors like
tenZeros = torch.zeros_like(input=oneToTen)
print(tenZeros)