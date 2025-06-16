import torch
import json
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from utils import learning_curve_pred
from model import CNNMLP
from load_data import pred_data_train


with open("../config.json", "r") as file:
    config = json.load(file)

# Load the saved CNN-LSTM model, modify it to extract features before the final layer, 
# generate features for training and validation data using the modified model, 
# and then train and evaluate an XGBoost model using these features.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dl, valid_dl = pred_data_train()

new_cnnmlp = CNNMLP(input_channel=1,output_channel=3, 
               num_cnn_layers=1,window_size=1, num_output_units=5).to(device)
new_cnnmlp.load_state_dict(torch.load(config["cnn_mlp_path"]))


class CNNMLPFeatureExtractor(nn.Module):
    def __init__(self, cnnmlp_model):
        super(CNNMLPFeatureExtractor, self).__init__()
        # get all layers except the last two layers
        self.feature_layers = cnnmlp_model.layers[:-1]

    def forward(self, x):
        return self.feature_layers(x)

feature_extractor = CNNMLPFeatureExtractor(new_cnnmlp).to(device)
feature_extractor.eval()

#  Initialize lists to store features and targets, then iterate through the training and validation DataLoaders 
# to extract features using the modified model and collect the corresponding target variables

train_features = []
train_targets = []
valid_features = []
valid_targets = []

with torch.no_grad():
    for X, y in train_dl:
        X= X.to(device)
        features = feature_extractor(X)
        train_features.append(features.cpu().numpy())
        train_targets.append(y.cpu().numpy())

    for X, y in valid_dl:
        X = X.to(device)
        features = feature_extractor(X)
        valid_features.append(features.cpu().numpy())
        valid_targets.append(y.cpu().numpy())

train_features = np.concatenate(train_features, axis=0)
train_targets = np.concatenate(train_targets, axis=0)
valid_features = np.concatenate(valid_features, axis=0)
valid_targets = np.concatenate(valid_targets, axis=0)

print("Training features shape:", train_features.shape)
print("Training targets shape:", train_targets.shape)
print("Validation features shape:", valid_features.shape)
print("Validation targets shape:", valid_targets.shape)

import xgboost as xgb

# Instantiate XGBRegressor

# Create classification matrices
dtrain_clf = xgb.DMatrix(train_features, train_targets)
dtest_clf = xgb.DMatrix(valid_features, valid_targets)
# Xgboost
params = {
    "objective": "multi:softprob", "tree_method": "hist","num_class": 5,
    "eval_metric": ["mlogloss","auc", "merror"]
}

n = 100
evals = [(dtest_clf, "validation"), (dtrain_clf, "train")]

results = xgb.train(
   params, dtrain_clf,
   num_boost_round=n,
   early_stopping_rounds=20,
   evals= evals,
   verbose_eval=10
)