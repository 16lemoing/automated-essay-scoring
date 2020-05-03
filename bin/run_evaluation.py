# -*- coding: utf-8 -*-

import sys
sys.path.append('../src/')
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})


from data import get_data, scan_essays, get_embedding_weights, get_encoded_data, EssayDataset, load_vocab
from models import Dense_NN
from train import predict_score

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

def get_set_scores(args,train_file):
    data = get_data(train_file, args.normalize_scores, args.set_idxs,
                    args.features, args.use_features, args.correct_spelling)
    _, _, _, _, set_scores = data
    return set_scores

def get_mean_length_and_features_of_set(nset,args,train_file,vocab):
    data = get_data(train_file, args.normalize_scores, args.set_idxs,
                    args.features, args.use_features, args.correct_spelling)
    essay_contents, essay_scores, essay_sets, essay_features, set_scores = data
    _, max_essay_len = scan_essays(essay_contents, args.remove_stopwords)
    essay_encoded, essay_lengths = get_encoded_data(essay_contents, essay_scores,
                                                    vocab, max_essay_len, args.remove_stopwords)
    essay_sets = np.array(essay_sets)
    essay_features = np.array(essay_features)
    essay_lengths = np.array(essay_lengths)
    in_set = np.argwhere(essay_sets == nset)
    set_features = essay_features[in_set]
    set_lengths = essay_lengths[in_set]
    return set_lengths.mean(), set_features.mean(axis=0).squeeze()

def get_simple_length_and_features(nset,args):
    length = 0.
    features = [0.]*(len(args.features)-1) + [float(nset)]
    return length,features

def get_word_tuples_dataset(word_tuples,nset,length,features,set_scores,args,scaler):
    N_words = len(word_tuples)
    n_words_in_tuple = len(word_tuples[0])

    lengths = np.array([length]*N_words)
    scores = np.array([0]*N_words)
    features = np.array([features]*N_words)
    sets = np.array([nset]*N_words)
    words_dataset = EssayDataset(word_tuples,lengths,scores,features,
                                sets,args.normalize_scores,
                                set_scores,args.use_features,args.scale_features)
    words_dataset.set_scaler(scaler)
    return words_dataset


def main(args):

    data_dir = Path("..") / "data"
    train_file = data_dir / args.train_file

    checkpoint_dir = Path("..") / "checkpoint" / args.checkpoint_name
    model_file = checkpoint_dir / args.model_name
    vocab_file = checkpoint_dir / args.vocab_name
    scaler_file = checkpoint_dir / args.scaler_name

    output_dir = Path('..') / 'outputs' / args.checkpoint_name
    output_dir.mkdir(parents=True,exist_ok=True)

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

    N_words = len(vocab.keys())
    N_sets = len(args.set_idxs)
    outputs = np.zeros((N_sets,N_words),dtype=np.float)
    for idx_of_set in range(N_sets):
        nset = args.set_idxs[idx_of_set]
        print(f'--- For set #{nset}')
        print('creating dataset of words')
        words = list(vocab.keys())
        encodings = list(vocab.values())
        word_tuples = list(map(lambda x:[x],encodings))
        set_scores = get_set_scores(args,train_file)
        length,features = get_mean_length_and_features_of_set(nset,args,train_file,vocab)
        # length,features = get_simple_length_and_features(nset,args)
        print(features)
        word_tuples_dataset = get_word_tuples_dataset(word_tuples, nset, length, features, set_scores,args,scaler)
        dataloader = DataLoader(word_tuples_dataset,batch_size=args.batch_size, num_workers = 5, shuffle = False)

        print('evaluating importance of words')
        outputs[idx_of_set,:] = predict_score(model,device,dataloader,True)

    words_orders = np.argsort(outputs)
    print('plotting results')
    plt.figure()
    for idx_of_set in range(N_sets):
        nset = args.set_idxs[idx_of_set]
        plt.plot(outputs[idx_of_set,words_orders[idx_of_set,:]][::-1],label=f'Set {nset}')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Words')
    plt.ylabel('Normalized score of one-word essay')
    plt.savefig(str(output_dir/'words_importance.pdf'),format='pdf')


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
