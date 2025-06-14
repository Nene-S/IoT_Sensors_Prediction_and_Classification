import torch.nn as nn
import torch

class CNNMLP(nn.Module):
  def __init__(self, input_channel=1,output_channel=3, 
               num_cnn_layers=1,window_size=1, num_output_units=5):
    super(CNNMLP, self).__init__()
    self.num_cnn_layers = num_cnn_layers
    self.window_size = window_size
  
    self.layers = self.CreateLayer(input_channel,output_channel, num_output_units)

  def forward(self, x):
    out = self.layers(x)

    return out

  def CreateLayer(self, input_channel,output_channel, num_output_units):
    layer = []
    for i in range(self.num_cnn_layers):
      layer.append(nn.Conv1d(input_channel, output_channel, kernel_size=1, padding=0, bias=False))
      layer.append(nn.BatchNorm1d(output_channel))
      layer.append(nn.ReLU())
      input_channel = output_channel
    layer.append(nn.Flatten())
    layer.append(nn.Linear(output_channel*self.window_size, num_output_units))

    return nn.Sequential(*layer)
  

class CNNLSTM(nn.Module):
  def __init__(self,input_size, lstm_hidden_units, cnn_output_channel,
               num_mote_ids, lstm_hidden_layer=1, embed_dim=2):
    super(CNNLSTM,self).__init__()
    self.lstm_input_size = cnn_output_channel + embed_dim
    self.lstm_hidden_units = lstm_hidden_units
    self.lstm_hidden_layer = lstm_hidden_layer
    self.mote_embed = nn.Embedding(num_mote_ids, embed_dim)
   
    self.cnn = nn.Conv1d(in_channels=input_size, out_channels=cnn_output_channel, 
                         kernel_size=3, padding=1, stride=1)
    self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=lstm_hidden_units, 
                        num_layers=lstm_hidden_layer, batch_first=True)
    
    self.fc = nn.Linear(lstm_hidden_units, 1)

  def forward(self,x_num, x_cat):
    batch_size = x_num.shape[0]

    x_num = x_num.permute(0,2,1)
    x_num = self.cnn(x_num)
    x_num = x_num.permute(0,2,1)

    mote_id_emb = self.mote_embed(x_cat[:, :, 0])
 
    combined_input = torch.cat((x_num, mote_id_emb), dim=2)

    h0, c0 = self.initial_hidden(batch_size)
    h0, c0 = h0.to(x_num.device), c0.to(x_num.device)

    out, (hidden, cell) = self.lstm(combined_input, (h0, c0))
    out = self.fc(out[:,-1, :])
    
    return out

  def initial_hidden(self, batch_size):
    hidden = torch.zeros(self.lstm_hidden_layer, batch_size, self.lstm_hidden_units)
    cell = torch.zeros(self.lstm_hidden_layer, batch_size, self.lstm_hidden_units)
    return hidden, cell
  


class CNNGRU(nn.Module):
  def __init__(self,input_size, gru_hidden_units, cnn_output_channel,
               num_mote_ids, gru_hidden_layer=1, embed_dim=2):
    super(CNNGRU,self).__init__()
    self.gru_input_size = cnn_output_channel + embed_dim
    self.gru_hidden_units = gru_hidden_units
    self.gru_hidden_layer = gru_hidden_layer
    self.mote_embed = nn.Embedding(num_mote_ids, embed_dim)
   
    self.cnn = nn.Conv1d(in_channels=input_size, out_channels=cnn_output_channel, 
                         kernel_size=3, padding=1, stride=1)
    self.lstm = nn.GRU(input_size=self.gru_input_size, hidden_size=gru_hidden_units, 
                        num_layers=gru_hidden_layer, batch_first=True)
    
    self.fc = nn.Linear(gru_hidden_units, 1)

  def forward(self,x_num, x_cat):
    batch_size = x_num.shape[0]

    x_num = x_num.permute(0,2,1)
    x_num = self.cnn(x_num)
    x_num = x_num.permute(0,2,1)

    mote_id_emb = self.mote_embed(x_cat[:, :, 0])
 
    combined_input = torch.cat((x_num, mote_id_emb), dim=2)

    h0 = self.initial_hidden(batch_size)
    h0= h0.to(x_num.device)

    out, hidden = self.lstm(combined_input, h0)
    out = self.fc(out[:,-1, :])
    
    return out

  def initial_hidden(self, batch_size):
    hidden = torch.zeros(self.lstm_hidden_layer, batch_size, self.lstm_hidden_units)
    return hidden


  
if __name__ == "__main__":
  rn = torch.rand(4, 1, 1)
  model = CNNMLP(1,2, num_cnn_layers=2, num_output_units=1, window_size=1)
  out = model(rn)
  print(out)
