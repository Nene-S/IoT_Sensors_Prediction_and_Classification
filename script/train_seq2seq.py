import torch 
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from load_data import pred_data_train
from seq2seq import *


def train(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fnc, device):
    encoder.train()
    decoder.train()
    total_loss = 0
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

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item() ** 0.5  # Taking RMSE 
    return total_loss / len(dataloader)


def valid(dataloader, encoder, decoder, loss_fnc, device):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            Xen_tem, Xen_cat, Xde_tem, Xdec_cat, y = data
            Xen_tem, Xen_cat, Xde_tem, Xdec_cat, y = Xen_tem.to(device), \
                Xen_cat.to(device), Xde_tem.to(device), Xdec_cat.to(device), y.to(device)

            encoder_outputs, en_hid, en_cell = encoder(Xen_tem, Xen_cat)
            decoder_outputs, _, _, _ = decoder(encoder_outputs, en_hid, en_cell, Xdec_cat, Xde_tem) 
            decoder_outputs = torch.stack(decoder_outputs, dim=2)

            loss = loss_fnc(decoder_outputs.view(-1), y.view(-1))
            total_loss += loss.item() ** 0.5  # Taking RMSE 

    return total_loss / len(dataloader)


def train_epochs(epochs, train_dl, valid_dl, encoder, decoder, 
                 encoder_optimizer, decoder_optimizer, loss_fnc, device):

    history = {
        "train_loss": [],
        "valid_loss": []
    }

    for epoch  in tqdm(range(1, epochs+1)):
        loss_train = train(train_dl, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fnc, device)
        loss_val = valid(valid_dl, encoder, decoder, loss_fnc, device)

        print(f"Epoch {epoch}: Train RMSE Loss = {loss_train:.4f} | Valid RMSE Loss = {loss_val:.4f}")

        history["train_loss"].append(loss_train)
        history["valid_loss"].append(loss_val)

    return history


def main():
    train_data, valid_data = pred_data_train()

    batch_size = 32
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    valid_dl = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.001
    epochs = 3

    encoder = Encoder(num_mote_ids=10, embed_dim=8, hidden_size=8, numeric_feat_size=1).to(device)
    decoder = AttenDecoder(hidden_size=8, embed_dim=8, num_mote_ids=10, numeric_feat_size=1).to(device)

    loss_fnc = nn.MSELoss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr)

    history = train_epochs(epochs, train_dl, valid_dl, encoder, decoder,
                           encoder_optimizer, decoder_optimizer, loss_fnc, device)

    # Plotting
    x_arr = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(13, 4))

    plt.plot(x_arr, history["train_loss"], "-o", label="Train RMSE Loss")
    plt.plot(x_arr, history["valid_loss"], "--<", label="Validation RMSE Loss")
    plt.xlabel("Epoch", size=15)
    plt.ylabel("Loss", size=15)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
