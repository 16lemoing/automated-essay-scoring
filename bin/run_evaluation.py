# -*- coding: utf-8 -*-

import sys
sys.path.append('../src/')
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import get_data, scan_essays, get_embedding_weights, get_encoded_data, EssayDataset, load_vocab
from models import Dense_NN
from train import evaluate_model

def get_train_dataset(args,train_file,vocab):
    data = get_data(train_file, args.normalize_scores, args.set_idxs,
                    args.features, args.use_features, args.correct_spelling)
    essay_contents, essay_scores, essay_sets, essay_features, set_scores = data
    _, max_essay_len = scan_essays(essay_contents, args.remove_stopwords)
    train_encoded, train_lengths = get_encoded_data(essay_contents, essay_scores,
                                                    vocab, max_essay_len, args.remove_stopwords)
    train_dataset = EssayDataset(train_encoded, train_lengths, essay_scores,
                                 essay_features, essay_sets, args.normalize_scores,
                                 set_scores, args.use_features, args.scale_features)
    return train_dataset

def get_train_mean_features(args,train_file,vocab):
    data = get_data(train_file, args.normalize_scores, args.set_idxs,
                    args.features, args.use_features, args.correct_spelling)
    essay_contents, essay_scores, essay_sets, essay_features, set_scores = data
    _, max_essay_len = scan_essays(essay_contents, args.remove_stopwords)
    train_encoded, train_lengths = get_encoded_data(essay_contents, essay_scores,
                                                    vocab, max_essay_len, args.remove_stopwords)

    N = len(train_lengths)
    def mean(l):
        return sum(map(lambda x:x/N,l))

    mean_length = mean(train_lengths)
    mean_score = mean(essay_scores)
    mean_features = list(np.array(essay_features).mean(axis=0))
    mean_set = args.set_idxs[np.argmax(np.bincount(essay_sets))]
    return mean_length,mean_score,mean_features,mean_set,set_scores


def main(args):

    data_dir = Path("..") / "data"
    train_file = data_dir / args.train_file

    checkpoint_dir = Path("..") / "checkpoint" / args.checkpoint_name
    model_file = checkpoint_dir / args.model_name
    vocab_file = checkpoint_dir / args.vocab_name
    scaler_file = checkpoint_dir / args.scaler_name

    print('loading vocabulary')
    vocab = load_vocab(vocab_file)

    print('loading model')
    device = torch.device(args.device)
    model,embedding_weights = Dense_NN.load(model_file,normalize_score=args.normalize_scores,return_embedding_weights=True)
    model = model.to(device)

    print('loading data scaler')
    if not scaler_file.exists():
        train_dataset = get_train_dataset(args,train_file,vocab)
        train_dataset.fit_scaler()
        train_dataset.save_scaler(scaler_file)
        scaler = train_dataset.get_scaler()
    else:
        scaler = EssayDataset.load_scaler(scaler_file)


    print('creating dataset of words')
    words = list(vocab.keys())
    encodings = list(vocab.values())
    N_words = len(encodings)
    encodings_2d = np.array(list(map(lambda x:[x],encodings)))
    mean_length,mean_score,mean_features,mean_set,set_score = get_train_mean_features(args,train_file,vocab)
    lengths = [mean_length]*N_words
    scores = np.around([mean_score]*N_words)
    features = [mean_features]*N_words
    sets = [mean_set]*N_words
    words_dataset = EssayDataset(encodings_2d,lengths,scores,features,
                                sets,args.normalize_scores,
                                set_score,args.use_features,args.scale_features)
    dataloader = DataLoader(words_dataset,batch_size=args.batch_size, num_workers = 5, shuffle = False)
    print('evaluating importance of words')
    output = evaluate_model(model,device,dataloader)
    print(output)

    # Fitting scaler
    # 1 Create Dataset
        # encoded_essays, essay_lengths, essay_scores, essay_features,
        # essay_sets, normalize_score, set_scores, use_features,
        # scale_features)
        # python3 run_test.py --name 'test_dense' --train_file 'train_x.tsv' --test_file 'test_x.tsv'
                # --dim 300 --remove_stopwords --normalize_scores
                # --correct_spelling --use_features --scale_features
                # --device 'cuda' --batch_size 256 --dropout 0.2
                # --hidden_size 300 128 --save_best_weights
    # 2 Feet dataset through model
    # 3 Collect output (note of each word)
    # 4 Order notes
    # 5 Match notes and decoded words
    # 6 Plots: curve of notes in the decreasing order, n-first words, n-last
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default = "train_x.tsv",
                        help = "name of .tsv file under which is saved train data")
    parser.add_argument('--checkpoint_name',default='checkpoint',
                        help = "name of the checkpoint directory")
    parser.add_argument('--model_name',default='fold0_weights.pth',
                        help = "name of the file where the weights are saved")
    parser.add_argument('--vocab_name',default='vocab.pkl',
                        help = "name of the file where the vocabulary is saved")
    parser.add_argument('--scaler_name',default='scaler.pkl',
                        help = "name of the file where the scaler is saved")
    parser.add_argument('--remove_stopwords', action = 'store_true',
                        help = "to ignore commonly used words")
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
    parser.add_argument('--batch_size', type = int, default = 256,
                        help = "size of batches to be processed in neural network model")

    args = parser.parse_args()
    main(args)
