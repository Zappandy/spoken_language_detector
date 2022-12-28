import torch
import copy
from torch.optim import Adam


def evaluation(model, val_data):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for spectro, label in val_data:
            pred = model(spectro)
            label_pred = torch.max(pred.data, 1)
            total += label.size(0)
            correct += (label_pred == label).sum().item()
    return correct / total * 100

class Trainer:

    def __init__(self, model, train_data, lr=1e-6):
        self.lr = lr
        self.optimizer = Adam
        self.loss_all_epochs = 0
        self.acc_all_epochs = 0
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_data = train_data
        self.model = copy.deepcopy(model)

    def train_loop(self, epochs=30):
        for epoch in tqdm(range(epochs)):
            pass
        pass

