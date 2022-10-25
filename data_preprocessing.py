import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

work_dir = "/media/andres/"  # complete path later and folders

train_data = # from dir
train_data = Subset(train_data, torch.arange(240)) # 80% 
test_data = Subset(test_data, torch.arange(60))  # 20%

train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)
