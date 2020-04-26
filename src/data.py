# -*- coding: utf-8 -*-

import re
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

def get_data(data_file, normalize_score, set_idxs, features, use_features, correct_spelling):
    data = pd.read_csv(data_file, sep = '\t', encoding = 'ISO-8859-1')
    essay_contents = data['corrected_essays'].values if correct_spelling else data['essay'].values
    essay_scores = data['domain1_score'].values
    essay_sets = data['essay_set'].values
    essay_features = data[features].values if use_features else [0] * len(essay_contents)
    filter_idx = [i for i, idx in enumerate(essay_sets) if idx in set_idxs]
    essay_contents = [essay_contents[i] for i in filter_idx]
    essay_scores = [essay_scores[i] for i in filter_idx]
    essay_sets = [essay_sets[i] for i in filter_idx]
    essay_features = [essay_features[i] for i in filter_idx]
    set_scores = get_set_scores(essay_scores, essay_sets)
    if normalize_score:
        essay_scores = normalize_scores(essay_scores, essay_sets, set_scores)
    return essay_contents, essay_scores, essay_sets, essay_features, set_scores

def scan_essays(essay_contents, remove_stopwords):
    words = []
    max_essay_len = 0
    for content in essay_contents:
        essay_words = tokenize_content(content, remove_stopwords)
        max_essay_len = max(len(essay_words), max_essay_len)
        words += essay_words
    single_words = set(words)
    vocab = {word: i + 1 for i, word in enumerate(single_words)}
    vocab[""] = 0
    return vocab, max_essay_len
    
def get_embedding_weights(vocab, embedding_dic, dim):
    """Get a matrix of embedding vectors from embedding dictionary (one for each word in vocab)"""
    weights = np.zeros((len(vocab), dim))
    words_found = 0
    missing_words = []
    for word in vocab:
        try: 
            weights[vocab[word]] = embedding_dic[word]
            words_found += 1
        except KeyError:
            missing_words.append(word)
            weights[vocab[word]] = np.random.normal(scale = 0.6, size = (dim))
    weights[0] *= 0
    print(f'found {words_found}/{len(vocab)} words, missing words are:')
    print(missing_words)
    return weights

def tokenize_content(essay_content, remove_stopwords, level="word"):
    """Break down the essay into words or sentences of words."""
    if level == "word":
        essay_content = re.sub("[^a-zA-Z]", " ", essay_content)
        words = essay_content.lower().split()
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        return words
    elif level == "sentence":
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        raw_sentences = tokenizer.tokenize(essay_content.strip())
        sentences = []
        for sentence in raw_sentences:
            if len(sentence) > 0:
                sentences.append(tokenize_content(sentence, remove_stopwords))
        return sentences
    
def get_encoded_data(essay_contents, essay_scores, vocab, max_essay_len, remove_stopwords):
    """Encode each essay based on vocabulary and pad to desired length for batch processing"""
    encoded_essays = np.zeros((len(essay_contents), max_essay_len))
    essay_lengths = np.zeros(len(essay_contents))
    for i, content in enumerate(essay_contents):
        words = tokenize_content(content, remove_stopwords)
        enc = np.array([vocab[word] for word in words])
        length = min(max_essay_len, len(enc))
        encoded_essays[i, :length] = enc[:length]
        essay_lengths[i] = length
    return encoded_essays, essay_lengths

def get_set_scores(essay_scores, essay_sets):
    set_scores = {i + 1:[] for i in range(8)}
    for score, set_id in zip(essay_scores, essay_sets):
        if not score in set_scores[set_id]:
            set_scores[set_id].append(score)
    for set_id in set_scores:
        set_scores[set_id] = sorted(set_scores[set_id])
    return set_scores

def normalize_scores(essay_scores, essay_sets, set_scores):
    normalized_scores = []
    for score, set_id in zip(essay_scores, essay_sets):
        normalized_scores.append((score - set_scores[set_id][0]) / (set_scores[set_id][-1] - set_scores[set_id][0]))
    return normalized_scores

def recover_scores(essay_scores, essay_sets, set_scores, round_to_known = False):
    recovered_scores = []
    for score, set_id in zip(essay_scores, essay_sets):
        recovered_score = score * (set_scores[set_id][-1] - set_scores[set_id][0]) + set_scores[set_id][0]
        if round_to_known:
            abs_diff = lambda value : abs(value - recovered_score)
            recovered_score = min(set_scores[set_id], key = abs_diff)
        recovered_scores.append(recovered_score)
    return recovered_scores

class EssayDataset(Dataset):

    def __init__(self, encoded_essays, essay_lengths, essay_scores, essay_features, essay_sets, normalize_score, set_scores, use_features, scale_features):
        self.encoded_essays = encoded_essays
        self.essay_lengths = essay_lengths
        self.essay_scores = essay_scores
        self.essay_features = essay_features
        self.essay_sets = essay_sets
        self.normalize_score = normalize_score
        self.set_scores = set_scores
        self.use_features = use_features
        self.scale_features = scale_features
        self.scaler = None
    
    def __len__(self):
        return len(self.encoded_essays)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.encoded_essays[idx]).long()
        lengths = torch.tensor(self.essay_lengths[idx]).long()
        scores = torch.tensor(self.essay_scores[idx]).float()
        feat = self.essay_features[idx]
        if self.scaler is not None:
            if len(feat.shape) == 1:
                feat = feat.reshape(1, -1)
            feat = self.scaler.transform(feat).squeeze()
        feat = torch.tensor(feat).float()
        return x, lengths, scores, feat
    
    def recover(self, scores, round_to_known = False):
        if self.normalize_score:
            recovered_scores = recover_scores(scores, self.essay_sets, self.set_scores, round_to_known)
        else:
            recovered_scores = scores
        return np.around(recovered_scores)
    
    def get_scores(self):
        return self.essay_scores
    
    def get_sets(self):
        return self.essay_sets
    
    def fit_scaler(self):
        if self.use_features and self.scale_features:
            self.scaler = StandardScaler()
            self.scaler.fit(self.essay_features)
    
    def get_scaler(self):
        return self.scaler
    
    def set_scaler(self, scaler):
        self.scaler = scaler