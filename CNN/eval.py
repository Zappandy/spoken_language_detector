from utils import evaluation
from model import CNNSpeechClassifier
#import torch.nn as nn
import torch
from torch.optim import Adam
from dataloader import SpeechDataset
from torch.utils.data import DataLoader


def load_components(checkpoint):


    epoch = checkpoint["epoch"]
    loss_fn = checkpoint["loss"]
    #loss_fn = nn.CrossEntropyLoss()

    model = CNNSpeechClassifier(channel_inputs=1, num_channels1=16,
                                num_channels2=32, kernel_size=2,
                                kernel_pool=2, padding=0, num_classes=3)


    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = Adam(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, epoch, loss_fn

def main():

    test_dir = "../Dataset/test/test"

    test_data = SpeechDataset(test_dir, "librosa")
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True)  # one is 4, 64, 862 - 4 despite batch size

    best_checkpoint = torch.load("model_output/best_speech_cnn.pth")
    final_checkpoint = torch.load("model_output/final_speech_cnn.pth")
    model, optimizer, epoch, loss_fn = load_components(best_checkpoint)

    acc, test_loss = evaluation(model, test_dataloader, loss_fn)

    print(f"Epoch {epoch+1}: test loss is {test_loss:.3f} | Accuracy is {acc:.2f}%")

if __name__ == "__main__":
    main()
