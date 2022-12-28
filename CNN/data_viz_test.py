# example, tst data has 540 files
import torch
f#rom dataloader import SpeechDataset
from model import CNNSpeechClassifier
#from train import ...
from torch.utils.data import DataLoader, Subset

test_dir = "../Dataset/test/test"
train_dir = "../Dataset/train/train"
test_data = SpeechDataset(test_dir, "librosa")
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

for spec, label in test_dataloader:
    print(type(spec))
    print(type(label))
    print(label)
    break

raise SystemExit

train_data = SpeechDataset(train_dir, "librosa")
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)


train_data = Subset(train_data, torch.arange(240)) # 80% 
test_data = Subset(test_data, torch.arange(60))  # 20%

train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)

# test visualizations

melspec_test = next(iter(test_dataloader))
print(melspec_test)
