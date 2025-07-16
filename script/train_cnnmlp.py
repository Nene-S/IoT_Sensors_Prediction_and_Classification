import torch
import json
import os
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from model import CNNMLP
from sklearn import model_selection 
from load_data import cls_data_train
from utils import learning_plot_cls

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
    if val_dl is None:
        return None, None
    
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
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    for epoch in tqdm(range(1, epochs + 1)):
        train_loss, train_acc = train(model, train_dl, loss_func, optimizer, device)
        val_loss, val_acc = valid(model, val_dl, loss_func, device)

        # Handle None from valid
        if val_loss is None:
            print(
                f"Epoch: {epoch:02d} |",
                f"train loss: {train_loss:.4f} | train accuracy: {train_acc:.4f}"
            )
        else:
            print(
                f"Epoch: {epoch:02d} |",
                f"train loss: {train_loss:.4f} | train accuracy: {train_acc:.4f} |",
                f"val loss: {val_loss:.4f} | validation accuracy: {val_acc:.4f}"
            )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    return history



def main():
    torch.manual_seed(42)

    train_dataset, val_dataset = cls_data_train()
    X, y = train_dataset.X ,train_dataset.y
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 1
    batch_size = 64
    folds = 5
    kf = model_selection.StratifiedKFold(n_splits=folds)
    for fold, (t_idx, v_idx) in enumerate(kf.split(X=X, y=y)):
        print(f"Fold {fold + 1}")
        print("-------")

        train_dl = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(t_idx))
        val_dl = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(v_idx) )

        lr = 1e-4
        model = CNNMLP(input_channel=1,output_channel=3,num_cnn_layers=1,
                   num_mlp_layers=3,window_size=5, num_output_units=5).to(device)
        # for p in model.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = train_epochs(model, epochs, train_dl, val_dl, loss_func, optimizer, device)
        torch.save(model.state_dict(), f"trained_models/cnn_mlp_{fold}.pth")
        path = f"figures/learning_plot_{fold}.png"
        learning_plot_cls(history, path, show=False)

        
    # Training on full train set
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    lr = 1e-4
    model = CNNMLP(input_channel=1,output_channel=3,num_cnn_layers=1,
                   num_mlp_layers=3,window_size=5, num_output_units=5).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    _= train_epochs(model, epochs, train_dl, None, loss_func, optimizer, device)

    torch.save(model.state_dict(), config["cnnmlp_model_path"])
    # learning_plot_cls(history, config["cls_lrn_plot_path"])


if __name__ == "__main__":
    main()


