import torch 
import json
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from utils import learning_curve_pred
from model import CNNLSTM, CNNGRU
from load_data import pred_data_train


with open("config.json", "r") as file:
    config = json.load(file)


def train(model, train_dl, optimizer, loss_fnc, mae, device):
    model.train()
    total_loss = 0
    total_mae = 0
    for X_num, X_cat, y in train_dl:
        optimizer.zero_grad()
        X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
        out = model(X_num, X_cat)
        loss = loss_fnc(out.view(-1), y.view(-1))
        cur_mae = mae(out.view(-1), y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mae += cur_mae.item() 

    avg_mse = total_loss/ len(train_dl)
    avg_rmse = avg_mse ** 0.5
    avg_mae = total_mae / len(train_dl) 

    return avg_mse, avg_rmse, avg_mae

def valid(model, valid_dl, loss_fnc, mae, device):
    model.eval()
    total_loss = 0
    total_mae = 0
    with torch.no_grad():
        for X_num, X_cat, y in valid_dl:
            X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
            out = model(X_num, X_cat)
            loss = loss_fnc(out.view(-1), y.view(-1))
            cur_mae = mae(out.view(-1), y.view(-1))

            total_loss += loss.item()
            total_mae += cur_mae.item() 

    avg_mse = total_loss/ len(valid_dl)
    avg_rmse = avg_mse ** 0.5
    avg_mae = total_mae / len(valid_dl) 

    return avg_mse, avg_rmse, avg_mae


def train_epochs(epochs, train_dl, valid_dl,model, optimizer,loss_fnc, mae, device):

    history = {
        "train_mse": [],
        "valid_mse": [],
        "train_rmse": [],
        "valid_rmse": [],
        "train_mae": [],
        "valid_mae": []
    }

    for epoch  in tqdm(range(1, epochs+1)):
        train_mse, train_rmse, train_mae = train(model, train_dl, optimizer, loss_fnc,mae,  device)
        val_mse, val_rmse, val_mae = valid(model, valid_dl,loss_fnc, mae, device)

        print(f"Epoch: {epoch} |", f"Train MSE: {train_mse:.4f} |", f"Valid MSE: {val_mse:.4f} |",
              f"Train RMSE: {train_rmse:.4f} |", f"Valid RMSE: {val_rmse:.4f} |",
              f"Train MAE: {train_mae:.4f} |", f"Valid MAE: {val_mae:.4f} |")

        history["train_mse"].append(train_mse)
        history["valid_mse"].append(val_mse)
        history["train_rmse"].append(train_rmse)
        history["valid_rmse"].append(val_rmse)
        history["train_mae"].append(train_mae)
        history["valid_mae"].append(val_mae)

    return history


def main():
    torch.manual_seed(42)
    train_data, valid_data = pred_data_train()

    batch_size = 64
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    valid_dl = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 1e-4
    epochs = 4
    loss_fnc = nn.MSELoss()
    mae = nn.L1Loss()

    #  Train CNN_LSTM
    lstm_model = CNNLSTM(input_size=1, lstm_hidden_units=8, cnn_output_channel=3,
               num_mote_ids=10, embed_dim=2).to(device)   
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr) 

    history = train_epochs(epochs, train_dl, valid_dl, lstm_model,lstm_optimizer, loss_fnc, mae, device)
    learning_curve_pred(history, config["cnnlstm_lrn_plot_path"])

    #  Train CNN_GRU
    gru_model = CNNGRU(input_size=1, gru_hidden_units=8, cnn_output_channel=3,
               num_mote_ids=10, gru_hidden_layer=1, embed_dim=2).to(device)
    gru_optimizer = torch.optim.Adam(gru_model.parameters(), lr)

    history = train_epochs(epochs, train_dl, valid_dl, gru_model,gru_optimizer, loss_fnc, mae, device)

    learning_curve_pred(history, config["cnngru_lrn_plot_path"])

    # torch.save()

if __name__ == "__main__":
    main()