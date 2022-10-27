import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split, Subset

# ls /media/andres/2D2DA2454B8413B5/test/test/ | grep -o '.....$' | uniq   
train_dir = "/media/andres/2D2DA2454B8413B5/train/train/"
test_dir = "/media/andres/2D2DA2454B8413B5/test/test/"

class SpeechDataset(Dataset):

    def __init__(self):
        numbers = [i for i in range(100)]
        self.data = numbers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

train_data = SpeechDataset()
dataloader = DataLoader(train_data, batch_size=10, shuffle=True)

for i, batch in enumerate(dataloader):
    print(i, batch)

raise SystemExit
train_data = Subset(train_data, torch.arange(240)) # 80% 
test_data = Subset(test_data, torch.arange(60))  # 20%

train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)
