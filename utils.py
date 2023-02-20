import torch
import matplotlib.pyplot as plt
from torch import cuda


def evaluation(model, val_data, loss_fn):

    device = 'cuda' if cuda.is_available() else 'cpu'

    model.eval()
    with torch.no_grad():
        correct = 0
        loss = 0
        total = 0
        for spectra, labels in val_data:

            spectra = spectra.unsqueeze(1)
            spectra = spectra.to(device)
            labels = labels.to(device)
            preds = model(spectra)
            vals, labels_preds = torch.max(preds.data, 1)  
            total += labels.size(0)  
            correct += (labels_preds == labels).sum().item()
            # loss
            err = loss_fn(preds, labels)
            loss += err.item()
        total_loss = loss / len(val_data)
    return correct / total * 100, total_loss

def visualize(epochs, tr_loss, val_loss, early_stop, save=False):

    fig = plt.figure()
    train, = plt.plot(torch.arange(epochs) + 1, tr_loss, '-og', label="Train")  
    valid, = plt.plot(torch.arange(epochs) + 1, val_loss, '-or', label="Valid")  
    if early_stop:
        pass
        # https://www.geeksforgeeks.org/plot-a-vertical-line-in-matplotlib/
    plt.xlabel('Epochs')
    plt.ylabel("Loss")    
    plt.legend(handles=[train, valid])
    plt.title('Loss over epochs')
    if save:
        fig.savefig("Loss_over_epochs.jpg", bbox_inches="tight", dpi=150)
    plt.close(fig)

def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'model_output/final_speech_cnn.pth')

plt.style.use('ggplot')  # for visualize!

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    #def __init__(self, best_valid_loss=float('inf')):
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        #self.path = "./model_output/best_speech_cnn.pth"
        self.path = "model_output/best_speech_cnn.pth"
        
    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion}, self.path)


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    # https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True