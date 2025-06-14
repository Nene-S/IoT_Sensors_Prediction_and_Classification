import torch
import json
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from model import CNNMLP
from load_data import cls_data_train
from utils import learning_curve_cls

with open("config.json", "r") as file:
    config = json.load(file)

def train(model, train_dl, loss_func, optimizer, device):
    model.train()
    train_loss = 0
    train_acc = 0
    for x_batch, y_batch in train_dl:
        optimizer.zero_grad()
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred = model(x_batch)
        loss = loss_func(pred, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        train_acc += is_correct.sum()

    return train_loss / len(train_dl) , train_acc / len(train_dl.dataset)


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

    return val_loss / len(val_dl) , val_acc / len(val_dl.dataset)


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
            f"Epoch: {epoch:02d} |",
            f"train loss: {train_loss:.4f} |", f"train_accuracy: {train_acc:.4f} |",
            f"val loss: {val_loss:.4f} |", f"val_accuracy: {val_acc:.4f}" 
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    return history


def main():
    torch.manual_seed(42)

    train_dataset, val_dataset = cls_data_train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    lr = 0.001
    model = CNNMLP(input_channel=1,output_channel=3, 
               num_cnn_layers=1,window_size=1, num_output_units=5).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    epochs = 3
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = train_epochs(model, epochs, train_dl, val_dl, loss_func, optimizer, device)

    learning_curve_cls(history, config["cls_lrn_plot_path"])

    # torch.save(model.state_dict(), config["cnn_mlp_path"])

if __name__ == "__main__":
    main()


