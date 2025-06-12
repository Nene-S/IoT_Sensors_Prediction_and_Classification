import torch
import json
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from model import CNNMLP
from load_data import cls_data_train
from utils import learning_curve

with open("config.json", "r") as file:
    config = json.load(file)

def train(model, train_dl, loss_func, optimizer, device):
    model.train()
    train_loss = 0
    train_acc = 0
    for x_batch, y_batch in train_dl:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred = model(x_batch)
        loss = loss_func(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        train_acc += is_correct.sum()

    return train_loss / len(train_dl) , train_acc / len(train_dl)


def valid(model, val_dl, loss_func, device):
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for x_batch, y_batch in val_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = loss_func(pred, y_batch)
            val_loss += loss.item()
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            val_acc += is_correct.sum()

    return val_loss / len(val_dl) , val_acc / len(val_dl)


def train_epochs(model, epochs, train_dl, val_dl, loss_func, optimizer, device):
    history = {
        "train_loss" : [],
        "train_acc" : [],
        "val_loss" : [],
        "val_acc" : []
        }
    
    for epoch in tqdm(range(1, epochs+1)):
        train_loss, train_acc = train(model, train_dl, loss_func, optimizer,device)
        val_loss, val_acc = valid(model, val_dl, loss_func,device)

        print(
            f"Epoch: {epoch} |",
            f"train loss: {train_loss} |", f"train_accuracy: {train_acc}",
            f"val loss: {val_loss} |", f"val_accuracy: {val_acc}" 
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    return history


def main():
    train_dataset, val_dataset = cls_data_train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    lr = 0.001
    model = CNNMLP(input_channel=1,output_channel=3, 
               num_cnn_layers=2,window_size=1, num_output_units=5).to(device)
    epochs = 3
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = train_epochs(model, epochs, train_dl, val_dl, loss_func, optimizer, device)

    learning_curve(history, config["cls_lrn_cur_path"])

if __name__ == "__main__":
    main()


