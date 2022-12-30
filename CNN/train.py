import copy
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from utils import evaluation, visualize, SaveBestModel

save_best_model = SaveBestModel()

class MyTrainer:

    def __init__(self, model, lr=1e-6):
        self.total_train_loss = []
        self.total_val_loss = []
        self.best_loss = float("inf")
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = copy.deepcopy(model)
        self.model.train()
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def train_loop(self, train_data, val_data, epochs=30, verbose=True, visual=False):
        for epoch in tqdm(range(epochs)):
            loss_curr_epoch = 0
            for spectra, labels in train_data:
                self.optimizer.zero_grad()

                spectra = spectra.unsqueeze(1)
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

            save_best_model(val_loss, epoch, self.model, self.optimizer, self.loss_fn)
            if acc > self.best_acc:
                #TODO: review saving
                #best_model = copy.deepcopy(self.model)
                #torch.save(self.model.state_dict(), 'model_best.pt')
                self.best_acc = acc
                continue


        
        if visual:
            visualize(epochs, self.total_train_loss, self.total_val_loss)
    
    def pretty_print(self, epoch, train_loss, val_loss, acc):
        print(f"Epoch {epoch+1}: train loss is {train_loss:.3f} | val loss is {val_loss:.3f} | Accuracy is {acc:.2f}%")



