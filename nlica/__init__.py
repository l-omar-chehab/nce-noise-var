import torch

# for two reasons:
# 1- scipy minimize yielded warnings in numerical precision
# 2- our experiments are sensitive to numerical precision
torch.set_default_dtype(torch.float64)  

