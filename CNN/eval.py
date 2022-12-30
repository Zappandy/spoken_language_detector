from utils import evaluation
from model import CNNSpeechClassifier
#import torch.nn as nn
import torch
from torch.optim import Adam
from dataloader import SpeechDataset
from torch.utils.data import DataLoader


def load_components(checkpoint, test_dataloader):


    epoch = checkpoint["epoch"]
    loss_fn = checkpoint["loss"]
    #loss_fn = nn.CrossEntropyLoss()

    CNN_model = CNNSpeechClassifier(channel_inputs=1, num_channels1=16,
                                    num_channels2=32, kernel_size=2,
                                    kernel_pool=2, padding=0, num_classes=3)


    CNN_model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = Adam(CNN_model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    acc, val_loss = evaluation(CNN_model, test_dataloader, loss_fn)

def main():

    test_dir = "../Dataset/test/test"

    test_data = SpeechDataset(test_dir, "librosa")
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True)  # one is 4, 64, 862 - 4 despite batch size

    best_checkpoint = torch.load("model_output/best_speech_cnn.pth")
    final_checkpoint = torch.load("model_output/final_speech_cnn.pth")
    load_components(best_checkpoint, test_dataloader)
    pass

if __name__ == "__main__":
    main()
