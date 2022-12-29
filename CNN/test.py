from train import evaluation
import torch.nn as nn


def main():

    loss_fn = nn.CrossEntropyLoss()
    #TODO: Load saved model
    acc, val_loss = evaluation(model, test_data, loss_fn)
    pass

if __name__ == "__main__":
    main()
