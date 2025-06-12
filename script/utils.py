import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os


class ClsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    
class PredDataset(Dataset):
    def __init__(self, X_en_num, X_en_cat, X_de_num,X_de_cat, y):
        self.X_en_num = X_en_num
        self.X_en_cat = X_en_cat
        self.X_de_num = X_de_num
        self.X_de_cat = X_de_cat
        self.y = y

    def __len__(self):
        return len(self.X_en_num)
    
    def __getitem__(self, index):
        return self.X_en_num[index], self.X_en_cat[index], self.X_de_num[index], \
            self.X_de_cat[index], self.y[index]
    

def learning_curve(hist, path):
        """Plots and saves the learning curve showing loss and accuracy over epochs."""

        x_arr = np.arange(len(hist["train_loss"])) + 1
        fig = plt.figure(figsize=(13, 4))

        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x_arr, hist["train_loss"], "-o", label="Train Loss")
        ax.plot(x_arr, hist["val_loss"], "--<", label="Validation Loss")
        ax.legend(fontsize=12)
        ax.set_xlabel("Epoch", size=15)
        ax.set_ylabel("Loss", size=15)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x_arr, hist["train_acc"], "-o", label="Train Accuracy")
        ax.plot(x_arr, hist["val_acc"], "--<", label="Validation Accuracy")
        ax.legend(fontsize=12)
        ax.set_xlabel("Epoch", size=15)
        ax.set_ylabel("Accuracy", size=15)

        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300)
        plt.show()