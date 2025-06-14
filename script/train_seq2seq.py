import torch 
import json
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from load_data import seq2seq_data_train
from utils import learning_curve_pred
from seq2seq import *

with open("config.json", "r") as file:
    config = json.load(file)


def train(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fnc, mae,  device):
    encoder.train()
    decoder.train()
    total_loss = 0
    total_mae = 0
    for data in dataloader:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        Xen_tem, Xen_cat, Xde_tem, Xdec_cat, y = data
        Xen_tem, Xen_cat, Xde_tem, Xdec_cat, y = Xen_tem.to(device), \
            Xen_cat.to(device), Xde_tem.to(device), Xdec_cat.to(device), y.to(device)

        encoder_outputs, en_hid, en_cell = encoder(Xen_tem, Xen_cat)
        decoder_outputs, _, _, _ = decoder(encoder_outputs, en_hid, en_cell, Xdec_cat, Xde_tem) 
        decoder_outputs = torch.stack(decoder_outputs, dim=2)

        loss = loss_fnc(decoder_outputs.view(-1), y.view(-1))
        cur_mae = mae(decoder_outputs.view(-1), y.view(-1))
        
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        total_mae += cur_mae.item() 

    avg_mse = total_loss/ len(dataloader)
    avg_rmse = avg_mse ** 0.5
    avg_mae = total_mae / len(dataloader) 

    return avg_mse, avg_rmse, avg_mae


def valid(dataloader, encoder, decoder, loss_fnc, mae, device):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    total_mae = 0
    with torch.no_grad():
        for data in dataloader:
            Xen_tem, Xen_cat, Xde_tem, Xdec_cat, y = data
            Xen_tem, Xen_cat, Xde_tem, Xdec_cat, y = Xen_tem.to(device), \
                Xen_cat.to(device), Xde_tem.to(device), Xdec_cat.to(device), y.to(device)

            encoder_outputs, en_hid, en_cell = encoder(Xen_tem, Xen_cat)
            decoder_outputs, _, _, _ = decoder(encoder_outputs, en_hid, en_cell, Xdec_cat, Xde_tem) 
            decoder_outputs = torch.stack(decoder_outputs, dim=2)

            loss = loss_fnc(decoder_outputs.view(-1), y.view(-1))
            cur_mae = mae(decoder_outputs.view(-1), y.view(-1))

            total_loss += loss.item()
            total_mae += cur_mae.item() 
    avg_mse = total_loss/ len(dataloader)
    avg_rmse = avg_mse ** 0.5
    avg_mae = total_mae / len(dataloader)         
     
    return avg_mse, avg_rmse, avg_mae


def train_epochs(epochs, train_dl, valid_dl, encoder, decoder, 
                 encoder_optimizer, decoder_optimizer, loss_fnc, mae, device):

    history = {
        "train_mse": [],
        "valid_mse": [],
        "train_rmse": [],
        "valid_rmse": [],
        "train_mae": [],
        "valid_mae": []
    }

    for epoch  in tqdm(range(1, epochs+1)):
        train_mse, train_rmse, train_mae = train(train_dl, encoder, decoder, encoder_optimizer, 
                                                 decoder_optimizer, loss_fnc,mae,  device)
        val_mse, val_rmse, val_mae = valid(valid_dl, encoder, decoder, loss_fnc, mae, device)

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
    train_data, valid_data = seq2seq_data_train()

    batch_size = 64
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    valid_dl = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 1e-4
    epochs = 5

    encoder = Encoder(num_mote_ids=10, embed_dim=8, hidden_size=8, numeric_feat_size=1).to(device)
    decoder = AttenDecoder(hidden_size=8, embed_dim=8, num_mote_ids=10, numeric_feat_size=1).to(device)

    loss_fnc = nn.MSELoss()
    mae = nn.L1Loss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr)

    history = train_epochs(epochs, train_dl, valid_dl, encoder, decoder,
                           encoder_optimizer, decoder_optimizer, loss_fnc, mae, device)

    learning_curve_pred(history, config["seq2seq_lrn_plot_path"])

    # torch.save()

if __name__ == "__main__":
    main()
