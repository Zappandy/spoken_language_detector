import torch
from dataloader import SpeechDataset, get_balanced_subset
from model import CNNSpeechClassifier2D
from train import MyTrainer
from torch.utils.data import DataLoader, Subset, random_split

SEED = 42
train_dir = "./Dataset/train/train"

train_data = SpeechDataset(train_dir, "librosa")

train_size = int(len(train_data) * 0.8)
val_size = len(train_data) - train_size

train_data, val_data = random_split(train_data, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=8, shuffle=True) 


print("TRAINING TIME")

kernel_size = (3, 3)
stride = (2, 2)
padding = (3, 3)
kernel_pool = 3
stride_pool = 2

CNN_model = CNNSpeechClassifier2D(channel_inputs=1, num_channels1=16,
                                num_channels2=32, num_channels3=64, num_channels4=128,
                                kernel_size=kernel_size, stride=stride,
                                kernel_pool=kernel_pool, stride_pool=stride_pool, padding=padding, num_classes=3)


trainer = MyTrainer(CNN_model)
trainer.train_loop(train_dataloader, val_dataloader, visual=True, epochs=10)
