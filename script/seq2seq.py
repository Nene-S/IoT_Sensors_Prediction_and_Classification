import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,num_mote_ids,embed_dim=4,hidden_size= 5, numeric_feat_size=1, num_layers=1): 
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_input_size = numeric_feat_size + (embed_dim)

        self.mote_embed = nn.Embedding(num_mote_ids, embed_dim)
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers,batch_first=True)

    def forward(self, x_num, x_cat):
    
        batch_size = x_num.shape[0]

        mote_id_emb = self.mote_embed(x_cat[:, :, 0])
        combined_input = torch.cat((x_num, mote_id_emb), dim=2)

        h0, c0 = self.initial_hidden(batch_size)
        h0, c0, = h0.to(x_num.device), c0.to(x_num.device)

        out, (hidden, cell) = self.lstm(combined_input, (h0, c0))

        return out, hidden, cell 
    
    def initial_hidden(self, batch_size):
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden, cell
    

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention,self).__init__()

        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):

        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        attention_weights = F.softmax(scores, dim=-1)

        context = torch.bmm(attention_weights, keys)

        return context, attention_weights
    

class AttenDecoder(nn.Module):
    def __init__(self, hidden_size, embed_dim, num_mote_ids, numeric_feat_size=1, num_layers=1):
        super(AttenDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.embed_size = embed_dim
        self.lstm_input_size = numeric_feat_size + embed_dim + hidden_size

        self.mote_embed = nn.Embedding(num_mote_ids, embed_dim)
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, en_outputs, en_hid,en_cell, x_cat, x_num=None, horizon=4):
        batch_size = en_outputs.size(0)
        mote_id_emb = self.mote_embed(x_cat[:, :, 0])
        
        dec_hid, dec_cell = en_hid,en_cell
        dec_input = torch.empty(batch_size, 1, dtype=torch.long, device=x_cat.device).fill_(0)
        dec_outputs = []
        attentions = []

        for i in range(horizon):
            if x_num != None:
                combined_input = torch.cat((x_num, mote_id_emb), dim=2)
                dec_input = combined_input[:, i, :].unsqueeze(1)
            else:
                dec_input = dec_outputs[-1].detach()

            dec_out, dec_hid, dec_cell, att_weights = self.forward_step(dec_input, dec_hid, dec_cell, en_outputs)
            dec_outputs.append(dec_out.detach())
            attentions.append(att_weights.detach())

        return dec_outputs, dec_hid, dec_cell, attentions
    
    def forward_step(self, input, hidden, cell, encoder_outputs):
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_lstm = torch.cat((input, context), dim=2)

        output, (hidden, cell) = self.lstm(input_lstm, (hidden, cell))
        output = self.fc(output)

        return output, hidden, cell, attn_weights



if __name__ == "__main__":
  x_tem = torch.rand(4, 6, 1) 
  x_cat = torch.randint(0, 10, (4, 6, 1))  
#   print(x_tem.shape, x_cat.shape) 
  encoder = Encoder(numeric_feat_size=1, num_mote_ids=10, embed_dim=2,hidden_size=2)
  decoder = AttenDecoder(hidden_size=2, embed_dim=2, num_mote_ids=10,numeric_feat_size=1)

  en_out, en_hid, en_cell = encoder(x_tem, x_cat)
  de_out, de_hid, de_cell, atten = decoder(en_out, en_hid, en_cell, x_cat, x_tem)
  print(de_out)

