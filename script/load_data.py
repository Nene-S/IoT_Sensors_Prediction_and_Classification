import torch
import numpy as np
import json
from utils import ClsDataset, PredDataset, Seq2SeqDataset

with open("config.json", "r") as file:
    config = json.load(file)

def cls_data_train():
    # load data and convert to tensor
    X_train= torch.tensor(np.load(config["cls_X_train_path"])).float()
    y_train= torch.tensor(np.load(config["cls_y_train_path"])).long()
    X_val= torch.tensor(np.load(config["cls_X_val_path"])).float()
    y_val= torch.tensor(np.load(config["cls_y_val_path"])).long()

    train_dataset = ClsDataset(X_train, y_train)
    val_dataset = ClsDataset(X_val, y_val)

    return train_dataset, val_dataset


def cls_data_test():
    X_test= torch.tensor(np.load(config["cls_X_test_path"])).float()
    y_test= torch.tensor(np.load(config["cls_y_test_path"])).long()
    
    test_dataset = ClsDataset(X_test, y_test)

    return test_dataset


def pred_data_train():
    X_num_train = torch.tensor(np.load(config["X_num_train_path"])).float()
    X_cat_train = torch.tensor(np.load(config["X_cat_train_path"])).long()
    y_train = torch.tensor(np.load(config["y_train_path"])).float()

    X_num_valid = torch.tensor(np.load(config["X_num_valid_path"])).float()
    X_cat_valid = torch.tensor(np.load(config["X_cat_valid_path"])).long()
    y_valid = torch.tensor(np.load(config["y_valid_path"])).float()

    train_dataset = PredDataset(X_num_train, X_cat_train, y_train)
    valid_dataset = PredDataset(X_num_valid, X_cat_valid, y_valid)

    return train_dataset, valid_dataset

def pred_data_test():
    X_num_test = torch.tensor(np.load(config["X_num_test_path"])).float()
    X_cat_test = torch.tensor(np.load(config["X_cat_test_path"])).long()
    y_test = torch.tensor(np.load(config["y_test_path"])).float()

    test_dataset = PredDataset(X_num_test, X_cat_test, y_test)

    return test_dataset


def seq2seq_data_train():
    X_enc_num_train = torch.tensor(np.load(config["X_enc_num_train_path"])).float()
    X_enc_cat_train = torch.tensor(np.load(config["X_enc_cat_train_path"])).long()
    X_dec_num_train = torch.tensor(np.load(config["X_dec_num_train_path"])).float()
    X_dec_cat_train = torch.tensor(np.load(config["X_dec_cat_train_path"])).long()
    y_dec_train     = torch.tensor(np.load(config["y_dec_train_path"])).float()

    X_enc_num_valid = torch.tensor(np.load(config["X_enc_num_valid_path"])).float()
    X_enc_cat_valid = torch.tensor(np.load(config["X_enc_cat_valid_path"])).long()
    X_dec_num_valid = torch.tensor(np.load(config["X_dec_num_valid_path"])).float()
    X_dec_cat_valid = torch.tensor(np.load(config["X_dec_cat_valid_path"])).long()
    y_dec_valid     = torch.tensor(np.load(config["y_dec_valid_path"])).float()

    train_dataset =Seq2SeqDataset(X_enc_num_train,X_enc_cat_train, X_dec_num_train, 
                               X_dec_cat_train, y_dec_train)
    val_dataset = Seq2SeqDataset(X_enc_num_valid, X_enc_cat_valid, 
                                X_dec_num_valid, X_dec_cat_valid, y_dec_valid)
    return train_dataset, val_dataset


def seq2seq_data_test():
  
    X_enc_num_test = torch.tensor(np.load(config["X_enc_num_test_path"])).float()
    X_enc_cat_test = torch.tensor(np.load(config["X_enc_cat_test_path"])).long()
    X_dec_num_test = torch.tensor(np.load(config["X_dec_num_test_path"])).float()
    X_dec_cat_test = torch.tensor(np.load(config["X_dec_cat_test_path"])).long()
    y_dec_test     = torch.tensor(np.load(config["y_dec_test_path"])).float()

    test_dataset = Seq2SeqDataset(X_enc_num_test, X_enc_cat_test,X_dec_num_test,
                               X_dec_cat_test, y_dec_test)
    return test_dataset

if __name__ =="__main__":
    _, _= pred_data_train()