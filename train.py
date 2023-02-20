import copy
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from torch import cuda
from utils import evaluation, visualize, save_model, EarlyStopping, SaveBestModel
from codecarbon import track_emissions


class MyTrainer:

    def __init__(self, model, lr=1e-6):

        self.device = 'cuda' if cuda.is_available() else 'cpu'

        self.total_train_loss = []
        self.total_val_loss = []
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = copy.deepcopy(model)
        self.model.to(self.device)
        self.model.train()
        lr = 1e-3  # 1e-4 best so far?. 1e-3 
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

    #@track_emissions(project_name="spoken_lang_detector", offline=True, country_iso_code='FRA')
    def train_loop(self, train_data, val_data, conv_type='2D', epochs=10, verbose=True, visual=False):

        #https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
        early_stopping = EarlyStopping()
        save_ckp = SaveBestModel()

        early_stop_value = None
        for epoch in tqdm(range(epochs)):
            loss_curr_epoch = 0
            for spectra, labels in train_data:
                self.optimizer.zero_grad()

                if conv_type == '2D':
                    spectra = spectra.unsqueeze(1)

                spectra = spectra.to(self.device)
                labels = labels.to(self.device)

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

            save_ckp(val_loss, epoch, self.model, self.optimizer, self.loss_fn)
            early_stopping(val_loss)


            if early_stopping.early_stop:
                early_stop_value = epoch+1
                print(f"Early stopping at epoch {early_stop_value}") 
        
        if visual:
            if not early_stop_value:
                early_stop_value = None
            visualize(epochs, self.total_train_loss, self.total_val_loss, early_stop_value)
        save_model(epochs=epochs, model=self.model, optimizer=self.optimizer, criterion=self.loss_fn)
    
    def pretty_print(self, epoch, train_loss, val_loss, acc):
        print(f"Epoch {epoch+1}: train loss is {train_loss:.3f} | val loss is {val_loss:.3f} | Accuracy is {acc:.2f}%")



