import torch.nn as nn
import torch

class CNNMLP(nn.Module):
  def __init__(self, input_channel=1,output_channel=3, 
               num_cnn_layers=1,window_size=5, num_output_units=5):
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
  def __inin__(self,input_size, horizon, lstm_hidden_units, cnn_output_channel, lstm_hidden_layer,
               num_mote_ids,num_fault_types,num_mote_fault,embed_dim= 32):
    super(CNNLSTM).__init__()

    self.lstm_hidden_units = lstm_hidden_units
    self.lstm_hidden_layer = lstm_hidden_layer
    self.mote_embed = nn.Embedding(num_mote_ids, embed_dim)
    self.fault_embed = nn.Embedding(num_fault_types, embed_dim)
    self.mote_fault_embed = nn.Embedding(num_mote_fault, embed_dim)

    self.cnn = nn.Conv1d(in_channels=input_size, out_channels=cnn_output_channel, 
                         kernel_size=3, padding=1, stride=1)
    self.lstm = nn.LSTM(input_size=cnn_output_channel, hidden_size=lstm_hidden_units, 
                        num_layers=lstm_hidden_layer, batch_first=True)
    
    self.fc = nn.Linear(lstm_hidden_units, horizon)

  def forward(self,x_num, x_cat):
    batch_size = x_num.shape[0]

    x_num = x_num.permute(0,2,1)
    x_num = self.cnn(x_num)
    x_num = x_num.permute(0,2,1)

    mote_id_emb = self.mote_embed(x_cat[:, :, 0])
    fault_type_emb = self.fault_embed(x_cat[:, :, 1])
    mote_fault_emb = self.mote_fault_embed(x_cat[:, :, 2])

    combined_input = torch.cat((x_num, mote_id_emb, fault_type_emb, mote_fault_emb), dim=2)

    h0, c0 = self.initial_hidden(batch_size)

    out, (hidden, cell) = self.lstm(combined_input, (h0, c0))
    out = self.fc(out[:,-1, :])
    
    return out

  def init_hidden(self, batch_size):
    hidden = torch.zeros(self.lstm_hidden_layer, batch_size, self.lstm_hidden_units)
    cell = torch.zeros(1, batch_size, self.lstm_hidden_units)
    return hidden, cell
  

  
if __name__ == "__main__":
  rn = torch.rand(4, 1, 3)
  model = CNNMLP(1,2, num_cnn_layers=2, num_output_units=1, window_size=3)
  # for params in model.parameters():
  #   print(params)
  out = model(rn)
  print(out)
