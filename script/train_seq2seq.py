import torch 
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from load_data import pred_data
from seq2seq import*



def train(dataloader, encoder, decoder,encoder_optimizer, decoder_optimizer, loss_fnc, device):
    total_loss = 0
    for data in dataloader:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        Xen_tem, Xen_cat, Xde_tem, Xdec_cat, y = data
        Xen_tem, Xen_cat, Xde_tem, Xdec_cat, y = Xen_tem.to(device), \
            Xen_cat.to(device), Xde_tem.to(device), Xdec_cat.to(device), y.to(device)

        encoder_outputs,en_hid, en_cell  = encoder(Xen_tem, Xen_cat)
        decoder_outputs, _, _, _ = decoder(encoder_outputs, en_hid, en_cell, Xde_tem, Xdec_cat) 

        loss = loss_fnc(decoder_outputs.view(-1), y.view(-1))

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
    return total_loss/ len(dataloader)


def valid(dataloader, encoder, decoder, loss_fnc,device):
    encoder.eval(), decoder.eval()
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            Xen_tem, Xen_cat, Xde_tem, Xdec_cat, y = data
            Xen_tem, Xen_cat, Xde_tem, Xdec_cat, y = Xen_tem.to(device), \
                Xen_cat.to(device), Xde_tem.to(device), Xdec_cat.to(device), y.to(device)

            encoder_outputs,en_hid, en_cell  = encoder(Xen_tem, Xen_cat)
            decoder_outputs, _, _, _ = decoder(encoder_outputs, en_hid, en_cell, Xde_tem, Xdec_cat) 

            loss = loss_fnc(decoder_outputs.view(-1), y.view(-1))

            total_loss += torch(loss.item **0.5)      
    return total_loss/ len(dataloader)


def train_epochs(epochs, train_dl, valid_dl, encoder, decoder, 
                 encoder_optimizer, decoder_optimizer, loss_fnc,device):

    history ={
        "train_loss" : [],
        "valid_loss" : []
    }
    for epoch in range(1, epochs+1):
        loss_train = train(train_dl, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fnc, device)
        loss_val = valid(valid_dl, encoder, decoder, loss_fnc, device)

        print(f"train RMSE loss: {loss_train} |", f"valid RMSE loss: {loss_val}")

        history["train_loss"].append(loss_train)
        history["valid_loss"].append(loss_val)
    
    return history


def main():
    train_data, valid_data = pred_data()

    batch_size = 32
    train_dl = DataLoader(train_data, batch_size, shuffle=False)
    valid_dl = DataLoader(valid_data, batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.001
    epochs = 30

    encoder = Encoder(num_mote_ids=10, embed_dim=8,hidden_size=8, numeric_feat_size=1).to(device)
    decoder = AttenDecoder(hidden_size=8, embed_dim=8, num_mote_ids=10,numeric_feat_size=1).to(device)

    loss_fnc = nn.MSELoss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr)

    history = train_epochs(epochs, train_dl, valid_dl, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fnc, device)

    # plot
    x_arr = np.arrange(len(history["train_plot"]) + 1)
    fig = plt.figure(figsize=(13, 4))

    plt.plot(x_arr, history["train_loss"], "-o", label="Train RMSE Loss")
    plt.plot(x_arr, history["val_loss"], "--<", label="Validation RSME Loss")
    plt.legend(fontsize=12)
    plt.set_xlabel("Epoch", size=15)
    plt.set_ylabel("Loss", size=15)
    plt.tight_layout()
    plt.show()


if __name__ =="__main__":
    main()
