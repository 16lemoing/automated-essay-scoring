# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

def create_embedding_layer(weights, non_trainable = False):
    num_embeddings, embedding_dim = weights.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer

class LSTM_NN(nn.Module):
    def __init__(self, weights, dim, normalize_score, use_features, extra_dim, dropout = 0, hidden_size = (10, 16), num_layers = 1, bidirectional = False, use_variable_length = False):
        super(LSTM_NN, self).__init__()
        self.use_features = use_features
        self.use_variable_length = use_variable_length
        self.embedding = create_embedding_layer(weights, True)
        self.lstm = nn.LSTM(input_size = dim, hidden_size = hidden_size[0], num_layers = num_layers, dropout = dropout, batch_first = True)
        self.fc_1 = nn.Linear(hidden_size[0] + extra_dim, hidden_size[1])
        self.dropout = nn.Dropout(p = dropout)
        self.fc_2 = nn.Linear(hidden_size[1], 1)
        self.activation_function = torch.sigmoid if normalize_score else torch.relu

    def forward(self, x, lengths, feat):
        x = self.embedding(x)
        if self.use_variable_length:
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, (h_n, c_n) = self.lstm(x)
        x = h_n[-1].squeeze()
        if self.use_features:
            x = torch.cat((x, feat), dim = 1)
        x = self.dropout(x)
        x = torch.relu(self.fc_1(x))
        x = self.dropout(x)
        x = self.activation_function(self.fc_2(x).squeeze())
        return x

class Dense_NN(nn.Module):
    def __init__(self, weights, dim, normalize_score, use_features, extra_dim, dropout = 0.2, hidden_size = (300, 16)):
        super(Dense_NN, self).__init__()
        self.use_features = use_features
        self.embedding = create_embedding_layer(weights, True)
        self.fc_1 = nn.Linear(dim + extra_dim, hidden_size[0])
        self.fc_2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc_3 = nn.Linear(hidden_size[1], 1)
        self.dropout = nn.Dropout(p = dropout)
        self.activation_function = torch.sigmoid if normalize_score else torch.relu

    def forward(self, x, lengths, feat):
        x = self.embedding(x)
        x = torch.mean(x, dim = 1)
        if self.use_features:
            x = torch.cat((x, feat), dim = 1)
        x = torch.relu(self.fc_1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc_2(x))
        x = self.dropout(x)
        x = self.activation_function(self.fc_3(x)).squeeze()
        return x

class Dense_feat_NN(nn.Module):
    def __init__(self, normalize_score, extra_dim, dropout = 0.2, hidden_size = (300, 16)):
        super(Dense_feat_NN, self).__init__()
        self.fc_1 = nn.Linear(extra_dim, hidden_size[0])
        self.fc_2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc_3 = nn.Linear(hidden_size[1], 1)
        self.dropout = nn.Dropout(p = dropout)
        self.activation_function = torch.sigmoid if normalize_score else torch.relu

    def forward(self, x, lengths, feat):
        x = feat
        x = torch.relu(self.fc_1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc_2(x))
        x = self.dropout(x)
        x = self.activation_function(self.fc_3(x)).squeeze()
        return x

class Convolutional_NN(nn.Module):
    def __init__(self,weights,dim,normalize_score,use_features,extra_dim,dropout=0.2,channel_sizes=[16,32,64]):
        super(Convolutional_NN,self).__init__()
        _, embedding_dim = weights.size()
        self.use_features = use_features
        self.embedding = create_embedding_layer(weights,True)
        self.c3 = nn.Conv2d(1,channel_sizes[0],(3,embedding_dim))
        self.c4 = nn.Conv2d(1,channel_sizes[1],(4,embedding_dim))
        self.c5 = nn.Conv2d(1,channel_sizes[2],(5,embedding_dim))
        self.dropout = nn.Dropout(p = dropout)
        self.fc = nn.Linear(sum(channel_sizes) + extra_dim,1)
        self.activation = torch.sigmoid if normalize_score else torch.relu

    def conv_block(self,x,conv_layer):
        x = conv_layer(x)
        x = torch.relu(x.squeeze(3))
        x = torch.max_pool1d(x,x.size()[-1]).squeeze(2)
        return x

    def forward(self,x,lengths,feat):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x3 = self.conv_block(x,self.c3)
        x4 = self.conv_block(x,self.c4)
        x5 = self.conv_block(x,self.c5)

        x = torch.cat((x3,x4,x5),1)
        if self.use_features:
            x = torch.cat((x,feat),1)
        x = self.dropout(x)
        x = self.fc(x).squeeze()
        return x
