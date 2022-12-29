import torch
import copy
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import Adam


def evaluation(model, val_data, loss_fn):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for spectra, labels in val_data:
            preds = model(spectra)
            labels_preds = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (labels_preds == labels).sum().item()
            # loss
            err = loss_fn(preds, labels)
            loss += err.item()
        total_loss = loss / len(val_data)
    return correct / total * 100, total_loss

class MyTrainer:

    def __init__(self, model, lr=1e-6):
        self.total_train_loss = []
        self.total_val_loss = []
        self.best_acc = 0
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = copy.deepcopy(model)
        self.model.train()
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def train_loop(self, train_data, val_data, epochs=30, verbose=True, visual=False):
        for epoch in tqdm(range(epochs)):
            loss_curr_epoch = 0
            for spectra, labels in train_data:
                self.optimizer.zero_grad()
                # rows=8, cols=64, channels=862?
                #print(spectra.shape)  # RuntimeError: Given groups=1, weight of size [16, 1, 2, 2], expected input[1, 8, 64, 862] to have 1 channels, but got 8 channels instead

                spectra = spectra.unsqueeze(1)
                #labels = labels.unsqueeze(0)
                #print(labels.shape)

                preds = self.model(spectra)  # 8, 3
                loss = self.loss_fn(preds, labels)
                loss.backward()
                self.optimizer.step()

                loss_curr_epoch += loss.item()

            train_loss = loss_curr_epoch / len(train_data)
            self.total_train_loss.append(train_loss)
            acc, val_loss = evaluation(self.model, val_data, self.loss_fn)
            self.total_val_loss.append(val_loss)
            if verbose:
                self.pretty_print(epoch=epoch, train_loss=train_loss, val_loss=val_loss, acc=acc)
                raise SystemExit
            if acc > self.best_acc:
                #TODO: Save model
                #torch.save
                continue
        
        if visual:
            self.visualize(epochs)
    
    def visualize(self, epochs, save=False):

        fig = plt.figure()
        train, = plt.plot(torch.arange(epochs) + 1, self.total_train_loss, '-og', label="Train")  
        valid, = plt.plot(torch.arange(epochs) + 1, self.total_val_loss, '-or', label="Valid")  
        plt.xlabel('Epochs')
        plt.ylabel("Loss")    
        plt.legend(handles=[train, valid])
        plt.title('Loss over epochs')
        if save:
            fig.savefig("Loss_over_epochs.jpg", bbox_inches="tight", dpi=150)
        #plt.show()
        plt.close(fig)

    def pretty_print(self, epoch, train_loss, val_loss, acc):
        print(f"Epoch {epoch+1}: train loss is {train_loss:.3f} | val loss is {val_loss:.3f} | Accuracy is {acc:.2f}%")



