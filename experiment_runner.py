# example, tst data has 540 files
import torch
from dataloader import SpeechDataset, get_balanced_subset
from model import CNNSpeechClassifier
from train import MyTrainer
from torch.utils.data import DataLoader, Subset, random_split

#torch.manual_seed(0)
SEED = 42
train_dir = "./Dataset/train/train"

train_data = SpeechDataset(train_dir, "librosa")


train_data = get_balanced_subset(train_data, 400) #4000 files for each language



train_size = int(len(train_data) * 0.8)
val_size = len(train_data) - train_size


train_data, val_data = random_split(train_data, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)  # 8, 64, 862 - 8
val_dataloader = DataLoader(val_data, batch_size=8, shuffle=True)  # 8, 64, 862 - 8

print("TRAINING TIME")

CNN_model = CNNSpeechClassifier(channel_inputs=1, num_channels1=16,
                                num_channels2=32, kernel_size=2,
                                kernel_pool=2, padding=0, num_classes=3)


trainer = MyTrainer(CNN_model)
trainer.train_loop(train_dataloader, val_dataloader, visual=True)