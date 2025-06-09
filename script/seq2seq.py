import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,numerical_input_features,num_mote_ids,num_fault_types,num_mote_fault,
                 embed_dim= 32,hidden_size= 5, num_layers= 2): 
        super(Encoder).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.embed_dim = embed_dim

        self.mote_embed = nn.Embedding(num_mote_ids, embed_dim)
        self.fault_embed = nn.Embedding(num_fault_types, embed_dim)
        self.mote_fault_embed = nn.Embedding(num_mote_fault, embed_dim)

        self.lstm_input_size = numerical_input_features + (3 * embed_dim)


        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers,batch_first=True, dropout=0.3)

    def forward(self, x_num, x_cat):
    
        batch_size = x_num.shape[0]

        mote_id_emb = self.mote_embed(x_cat[:, :, 0])
        fault_type_emb = self.fault_embed(x_cat[:, :, 1])
        mote_fault_emb = self.mote_fault_embed(x_cat[:, :, 2])

        combined_input = torch.cat((x_num, mote_id_emb, fault_type_emb, mote_fault_emb), dim=2)

        h0, c0 = self.initial_hidden(batch_size)

        out, (hidden, cell) = self.lstm(combined_input, (h0, c0))

        return out, hidden, cell 
    
    def initial_hidden(self, batch_size):
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden, cell
    

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size=5, num_layers=2, mote_id_d=10, fault_type_d=5, mote_fault_d=10,emb_size=32):
        super(Decoder).__init__()
        self.hidden_size = hidden_size
   
        self.mote_embed = nn.Embedding(mote_id_d, emb_size)
        self.fault_embed = nn.Embedding(fault_type_d, emb_size)
        self.mote_fault_embed = nn.Embedding(mote_fault_d, emb_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, horizon, hidden, cell,  dec_x, mote_id_cat, fault_type_cat, mote_fault_cat,):
        mote_id_emb = self.mote_embed(mote_id_cat)
        fault_typet_emb = self.mote_embed(fault_type_cat)
        mote_fault_emb = self.mote_embed(mote_fault_cat)
        x = torch.cat((dec_x, mote_id_emb,fault_typet_emb, mote_fault_emb), dim=1)
        output = []
        input = []

        for i in horizon:
            out = self.lstm(x[i], (hidden, cell))
            output.append(out)



class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):

        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        attention_weights = F.softmax(scores, dim=-1)

        context = torch.bmm(attention_weights, keys)

        return context, attention_weights

    


