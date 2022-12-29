# example, tst data has 540 files
import torch
import torch.nn as nn
from dataloader import SpeechDataset
from model import CNNSpeechClassifier
from train import MyTrainer
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim import Adam

#torch.manual_seed(0)
SEED = 42

test_dir = "../Dataset/test/test"
train_dir = "../Dataset/train/train"

train_data = SpeechDataset(train_dir, "librosa")

# FOR TEST PURPOSES REMOVE AFTER
train_data = Subset(train_data, torch.arange(100))

test_data = SpeechDataset(test_dir, "librosa")
train_size = int(len(train_data) * 0.8)
val_size = len(train_data) - train_size


train_data, val_data = random_split(train_data, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)  # 8, 64, 862 - 8
val_dataloader = DataLoader(val_data, batch_size=8, shuffle=True)  # 8, 64, 862 - 8
# not an issue with 4 batch, but others end up with 2 sizes.
test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True)  # one is 4, 64, 862 - 4 despite batch size
#[1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 27, 30, 36, 45, 54, 60]

print("TRAINING TIME")
#channel_inputs = 1
#num_channels1 = 16
#num_channels2 = 16
#kernel_size = 2
#kernel_pool = 2
#padding = 0 
#num_classes = 3
#cnn_layer1 = nn.Sequential(nn.Conv2d(channel_inputs, num_channels1, kernel_size=kernel_size, padding=padding),
#                           nn.ReLU(),
#                           nn.BatchNorm2d(num_channels1),
#                           nn.MaxPool2d(kernel_pool))
#
#cnn_layer2 = nn.Sequential(nn.Conv2d(num_channels1, num_channels2, kernel_size=kernel_size, padding=padding),
#                           nn.ReLU(),
#                           nn.BatchNorm2d(num_channels2),
#                           nn.MaxPool2d(kernel_pool))
#fc_layer = nn.Linear(num_channels2*15*214, num_classes)  # shape of cnn_layer 2 after convolution of image!



CNN_model = CNNSpeechClassifier(channel_inputs=1, num_channels1=16,
                                num_channels2=32, kernel_size=2,
                                kernel_pool=2, padding=0, num_classes=3)

#optimizer = Adam(CNN_model.parameters(), lr=1e-6)
#data_bit = iter(train_dataloader)
#mel, lab = next(data_bit)
#print(mel.shape)
#new_mel = mel.unsqueeze(1)
#print(mel.shape)
#x = cnn_layer1(new_mel)
#print(x.shape)
#y = cnn_layer2(x)
#print(y.shape)
#z = y.reshape(y.shape[0], -1)
#w = fc_layer(z)
#print(w.shape)
#print(lab.shape)
#c_loss = nn.CrossEntropyLoss()
#loss = c_loss(w, lab)
#loss.backward()
#optimizer.step()
#print(f"{loss:.2f}")
#raise SystemExit

trainer = MyTrainer(CNN_model)
trainer.train_loop(train_dataloader, val_dataloader)