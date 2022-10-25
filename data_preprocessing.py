import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

work_dir = "/media/andres/"  # complete path later and folders

train_data = # from dir
train_data = Subset(train_data, torch.arange(240))
test_data = Subset(test_data, torch.arange(60))

train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
