from utils import evaluation
from model import CNNSpeechClassifier2D
from dataloader import get_balanced_subset
import torch
from torch import cuda
from torch.optim import AdamW
from dataloader import SpeechDataset
from torch.utils.data import DataLoader


device = 'cuda' if cuda.is_available() else 'cpu'

def load_components(checkpoint):


    epoch = checkpoint["epoch"]
    loss_fn = checkpoint["loss"]

    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (3, 3)
    kernel_pool = 3
    stride_pool = 2
    
    model = CNNSpeechClassifier2D(channel_inputs=1, num_channels1=16,
                                num_channels2=32, num_channels3=64, num_channels4=128,
                                kernel_size=kernel_size, stride=stride,
                                kernel_pool=kernel_pool, stride_pool=stride_pool, padding=padding, num_classes=3)
    

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = AdamW(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    model.to(device)

    return model, optimizer, epoch, loss_fn

def main():

    test_dir = "Dataset/test/test"

    test_data = SpeechDataset(test_dir, "librosa")
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True)

    best_checkpoint = torch.load("model_output/best_speech_cnn.pth", map_location=device)  # weird form of early stopping. Should add patience
    final_checkpoint = torch.load("model_output/final_speech_cnn.pth", map_location=device)
    model, optimizer, epoch, loss_fn = load_components(best_checkpoint)

    acc, test_loss = evaluation(model, test_dataloader, loss_fn)

    print(f"Best epoch {epoch}: test loss is {test_loss:.3f} | Accuracy is {acc:.2f}%")

if __name__ == "__main__":
    main()
