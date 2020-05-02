# -*- coding: utf-8 -*-

import sys
sys.path.append("../src")
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader

from tools import Logger
from glove import get_glove
from word2vec import get_word2vec
from data import get_data, scan_essays, get_embedding_weights, get_encoded_data, EssayDataset
from models import Dense_NN, Dense_feat_NN, LSTM_NN, Convolutional_NN
from train import train_model

def main(args):

    data_dir = Path("..") / "data"
    glove_dir = data_dir / 'glove'
    log_dir = Path("..") / "log"
    logger = Logger(log_dir, "simulation", args)
    train_file = data_dir / args.train_file

    print("loading data")
    data = get_data(train_file, args.normalize_scores, args.set_idxs,
                    args.features, args.use_features, args.correct_spelling)
    essay_contents, essay_scores, essay_sets, essay_features, set_scores = data
    vocab, max_essay_len = scan_essays(essay_contents, args.remove_stopwords)

    print(f"preparing {args.embedding_type} embedding")
    if args.embedding_type == "word2vec":
        word2vec = get_word2vec(essay_contents, args.remove_stopwords, args.dim)
        embedding_weights = get_embedding_weights(vocab, word2vec, args.dim)
    elif args.embedding_type == "glove":
        glove = get_glove(glove_dir, args.glove_type, args.dim)
        embedding_weights = get_embedding_weights(vocab, glove, args.dim)
    elif args.embedding_type == "random":
        embedding_weights = get_embedding_weights(vocab, {}, args.dim)

    print("training using cross validation")
    cross_validation_folds = KFold(n_splits = 5, shuffle = True)
    for count, (train_idx, valid_idx) in enumerate(cross_validation_folds.split(essay_contents)):
        print(f"--------fold {count + 1}--------")
        logger.init_fold()
        train_contents = [essay_contents[i] for i in train_idx]
        train_scores = np.array([essay_scores[i] for i in train_idx])
        train_sets = [essay_sets[i] for i in train_idx]
        train_features = [essay_features[i] for i in train_idx]
        valid_contents = [essay_contents[i] for i in valid_idx]
        valid_scores = np.array([essay_scores[i] for i in valid_idx])
        valid_sets = [essay_sets[i] for i in valid_idx]
        valid_features = [essay_features[i] for i in valid_idx]

        device = torch.device(args.device)

        train_encoded, train_lengths = get_encoded_data(train_contents, train_scores,
                                                        vocab, max_essay_len, args.remove_stopwords)
        valid_encoded, valid_lengths = get_encoded_data(valid_contents, valid_scores,
                                                        vocab, max_essay_len, args.remove_stopwords)
        train_dataset = EssayDataset(train_encoded, train_lengths, train_scores,
                                     train_features, train_sets, args.normalize_scores,
                                     set_scores, args.use_features, args.scale_features)
        valid_dataset = EssayDataset(valid_encoded, valid_lengths, valid_scores,
                                     valid_features, valid_sets, args.normalize_scores,
                                     set_scores, args.use_features, args.scale_features)
        train_dataset.fit_scaler()
        valid_dataset.set_scaler(train_dataset.get_scaler())

        extra_dim = len(args.features) if args.use_features else 0
        if args.model_type == "dense":
            model = Dense_NN(torch.tensor(embedding_weights), args.dim, args.normalize_scores,
                             args.use_features, extra_dim, args.dropout, args.hidden_size).to(device)
        elif args.model_type == "dense_feat":
            model = Dense_feat_NN(args.normalize_scores, extra_dim, args.dropout, args.hidden_size).to(device)
        elif args.model_type == "lstm":
            model = LSTM_NN(torch.tensor(embedding_weights), args.dim, args.normalize_scores,
                            args.use_features, extra_dim, args.dropout, args.hidden_size,
                            args.num_layers, args.is_bidirectional, args.use_variable_length).to(device)
        elif args.model_type == 'conv':
            model = Convolutional_NN(torch.tensor(embedding_weights),args.dim,args.normalize_scores,
                            args.use_features,extra_dim,args.dropout,args.channel_sizes).to(device)

        train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers = 5, shuffle = True)
        valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, num_workers = 5, shuffle = False)

        train_model(model, device, args.lr, args.epochs, train_dataloader, valid_dataloader, logger)

    print(f"saving results to {log_dir} (id: {logger.id})")
    logger.save_plots()
    logger.update_csv()

    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required = True,
                        help = "short description of what is being simulated")
    parser.add_argument('--train_file', default = "train_x.tsv",
                        help = "name of .tsv file under which is saved train data")
    parser.add_argument('--dim', type = int, default = 50,
                        help = "dimension of embedding vectors")
    parser.add_argument('--remove_stopwords', action = 'store_true',
                        help = "to ignore commonly used words")
    parser.add_argument('--embedding_type', default = 'word2vec',
                        help = "any of ['glove', 'word2vec', 'random']")
    parser.add_argument('--glove_type', default = '6B',
                        help = "any of ['42B.300d', '840B.300d', '6B', 'twitter.27B']"\
                        "('bin/prepare_glove.py' should have been executed for this type beforehand)")
    parser.add_argument('--normalize_scores', action = 'store_true',
                        help = "to rescale scores to range [0, 1]")
    parser.add_argument('--set_idxs', type = int, nargs = '+', default = [1, 2, 3, 4, 5, 6, 7, 8],
                        help = "select essays corresponding to set idxs (each set is a different task)")
    parser.add_argument('--correct_spelling', action = 'store_true',
                        help = "to use corrected essays instead of raw essays "\
                        "('bin/compute_x_features.py' should have been executed beforehand)")
    parser.add_argument('--features', nargs = '+',
                        default = ['corrections', 'token_count', 'unique_token_count', 'nostop_count',
                                   'sent_count', 'ner_count', 'comma', 'question', 'exclamation',
                                   'quotation', 'organization', 'caps', 'person', 'location', 'money',
                                   'time', 'date', 'percent', 'noun', 'adj', 'pron', 'verb', 'noun',
                                   'cconj', 'adv', 'det', 'propn', 'num', 'part', 'intj', 'essay_set'],
                        help = "list of extra features")
    parser.add_argument('--use_features', action = 'store_true',
                        help = "to use extra features when predicting essay score "\
                        "('bin/compute_x_features.py' should have been executed beforehand)")
    parser.add_argument('--scale_features', action = 'store_true',
                        help = "to scale extra features so they have 0 mean and unit variance")
    parser.add_argument('--device', default = 'cpu',
                        help = "any of ['cpu', 'cuda']")
    parser.add_argument('--model_type', default = 'dense',
                        help = "any of ['dense', 'dense_feat', 'lstm', 'conv'] (neural network model)")
    parser.add_argument('--batch_size', type = int, default = 64,
                        help = "size of batches to be processed in neural network model")
    parser.add_argument('--lr', type = float, default = 0.01,
                        help = "learning rate for stochastic gradient descent")
    parser.add_argument('--epochs', type = int, default = 50,
                        help = "number of epochs to run")
    parser.add_argument('--dropout', type = float, default = 0.0,
                        help = "dropout for neural network")
    parser.add_argument('--hidden_size', type = int, nargs = '+', default = [10, 16],
                        help = "hidden size for neural network layers")
    parser.add_argument('--num_layers', type = int, default = 1,
                        help = "number of layers for neural network")
    parser.add_argument('--is_bidirectional', action = 'store_true',
                        help = "to have a bidirectional lstm model")
    parser.add_argument('--use_variable_length', action = 'store_true',
                        help = "to take into account length of sequential input")
    parser.add_argument('--channel_sizes',type = int, nargs = '+', default = [16,32,64],
                        help = 'number of channels of the convolutional network\'s layers')

    args = parser.parse_args()
    main(args)
