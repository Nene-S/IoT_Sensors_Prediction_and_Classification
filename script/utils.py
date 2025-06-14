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
    def __init__(self, X_num, X_cat, y):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y

    def __len__(self):
        return len(self.X_num)
    
    def __getitem__(self, index):
        return self.X_num[index], self.X_cat[index], self.y[index]
    
    
class Seq2SeqDataset(Dataset):
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
    

def learning_plot_cls(hist, path):
        """Plots and saves the learning curve showing loss and accuracy over epochs."""

        x_arr = np.arange(len(hist["train_loss"])) + 1
        fig = plt.figure(figsize=(11, 4))

        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x_arr, hist["train_loss"], "-o", label="Train Loss")
        ax.plot(x_arr, hist["val_loss"], "-o", label="Validation Loss")
        ax.legend(fontsize=10)
        ax.set_xlabel("Epoch", size=15)
        ax.set_ylabel("Loss", size=15)
        

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x_arr, hist["train_acc"], "-o", label="Train Accuracy")
        ax.plot(x_arr, hist["val_acc"], "-o", label="Validation Accuracy")
        ax.legend(fontsize=10)
        ax.set_xlabel("Epoch", size=12)
        ax.set_ylabel("Accuracy", size=12)
      

        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300)
        plt.show()


def learning_curve_pred(hist, path):
    """Plots and saves learning curves showing MSE, RMSE, and MAE over epochs."""

    x_arr = np.arange(1, len(hist["train_mse"]) + 1)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

   
    axs[0].plot(x_arr, hist["train_mse"], "-o", label="Train MSE")
    axs[0].plot(x_arr, hist["valid_mse"], "-s", label="Validation MSE")
    axs[0].set_title("Mean Squared Error", fontsize=14)
    axs[0].set_xlabel("Epoch", fontsize=12)
    axs[0].set_ylabel("MSE", fontsize=12)
    axs[0].legend()

    axs[1].plot(x_arr, hist["train_rmse"], "-o", label="Train RMSE")
    axs[1].plot(x_arr, hist["valid_rmse"], "-s", label="Validation RMSE")
    axs[1].set_title("Root Mean Squared Error", fontsize=14)
    axs[1].set_xlabel("Epoch", fontsize=12)
    axs[1].set_ylabel("RMSE", fontsize=12)
    axs[1].legend()
    
    axs[2].plot(x_arr, hist["train_mae"], "-o", label="Train MAE")
    axs[2].plot(x_arr, hist["valid_mae"], "-s", label="Validation MAE")
    axs[2].set_title("Mean Absolute Error", fontsize=14)
    axs[2].set_xlabel("Epoch", fontsize=12)
    axs[2].set_ylabel("MAE", fontsize=12)
    axs[2].legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.show()