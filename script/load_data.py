import torch
import numpy as np
import json
from utils import CustomDataset

with open("config.json", "r") as file:
    config = json.load(file)

def cls_data():
    # load data array
    cls_X_train= np.load(config["cls_X_train_path"])
    cls_y_train= np.load(config["cls_y_train_path"])
    cls_X_val= np.load(config["cls_X_val_path"])
    cls_y_val= np.load(config["cls_y_val_path"])

    # convert to tensor
    X_train = torch.tensor(cls_X_train).float()
    y_train = torch.tensor(cls_y_train).float()
    X_val = torch.tensor(cls_X_val).float()
    y_val = torch.tensor(cls_y_val).float()

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)

    return train_dataset, val_dataset

def pred_data():
    # load data array
    pred_X_train= np.load(config["pred_X_train_path"])
    pred_y_train= np.load(config["pred_y_train_path"])
    pred_X_val= np.load(config["pred_X_val_path"])
    pred_y_val= np.load(config["pred_y_val_path"])

    # convert to tensor
    X_train = torch.tensor(pred_X_train).float()
    y_train = torch.tensor(pred_y_train).float()
    X_val = torch.tensor(pred_X_val).float()
    y_val = torch.tensor(pred_y_val).float()

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)

    return train_dataset, val_dataset