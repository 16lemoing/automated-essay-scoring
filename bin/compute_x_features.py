# -*- coding: utf-8 -*-

import sys
sys.path.append("../src")
import argparse
from pathlib import Path
import pandas as pd
from spellchecker import SpellChecker
from nltk.tokenize.treebank import TreebankWordDetokenizer
from string import punctuation
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm

from glove import get_glove

def main(args):
    
    data_dir = Path("..") / "data"
    data_file = data_dir / args.data_file
    out_file = data_dir / args.out_file
    glove_dir = data_dir / "glove"
    
    print("loading data")
    data = pd.read_csv(data_file, sep='\t', encoding='ISO-8859-1')
    essay_contents = data['essay'].values

    print("loading glove")
    glove = get_glove(glove_dir, args.glove_type, args.dim)

    print("correcting spelling errors in essays and computing basic features")
    spell = SpellChecker()
    detokenizer = TreebankWordDetokenizer()
    sents = []
    tokens = []
    lemma = []
    pos = []
    ner = []
    corrected_essay_contents = []
    corrections_num = []
    nlp = spacy.load("en_core_web_sm")
    stop_words = set(STOP_WORDS)
    stop_words.update(punctuation)
    for essay in tqdm(nlp.pipe(data['essay'], batch_size = 100, n_threads = 10), total = len(essay_contents)):
        if essay.is_parsed:
            words = [e.text for e in essay]
            spelling_erros = 0
            for i, word in enumerate(words):
                if not word in glove and not '@' in word:
                    correction = spell.correction(word)
                    if correction != word:
                        spelling_erros += 1
                        words[i] = correction
            corrected_essay_contents.append(detokenizer.detokenize(words))
            corrections_num.append(spelling_erros)
            tokens.append(words)
            sents.append([sent.string.strip() for sent in essay.sents])
            pos.append([e.pos_ for e in essay])
            ner.append([e.text for e in essay.ents])
            lemma.append([n.lemma_ for n in essay])
        else:
            corrected_essay_contents.append(None)
            corrections_num.append(None)
            tokens.append(None)
            lemma.append(None)
            pos.append(None)
            sents.append(None)
            ner.append(None)
    
    print("computing extra features")
    data['corrected_essays'] = corrected_essay_contents
    data['corrections'] = corrections_num
    data['tokens'] = tokens
    data['lemma'] = lemma
    data['pos'] = pos
    data['sents'] = sents
    data['ner'] = ner
    data['token_count'] = data.apply(lambda x: len(x['tokens']), axis=1)
    data['unique_token_count'] = data.apply(lambda x: len(set(x['tokens'])), axis=1)
    data['nostop_count'] = data.apply(lambda x: len([token for token in x['tokens'] if token not in stop_words]), axis=1)
    data['sent_count'] = data.apply(lambda x: len(x['sents']), axis=1)
    data['ner_count'] = data.apply(lambda x: len(x['ner']), axis=1)
    data['comma'] = data.apply(lambda x: x['corrected_essays'].count(','), axis=1)
    data['question'] = data.apply(lambda x: x['corrected_essays'].count('?'), axis=1)
    data['exclamation'] = data.apply(lambda x: x['corrected_essays'].count('!'), axis=1)
    data['quotation'] = data.apply(lambda x: x['corrected_essays'].count('"') + x['corrected_essays'].count("'"), axis=1)
    data['organization'] = data.apply(lambda x: x['corrected_essays'].count(r'@ORGANIZATION'), axis=1)
    data['caps'] = data.apply(lambda x: x['corrected_essays'].count(r'@CAPS'), axis=1)
    data['person'] = data.apply(lambda x: x['corrected_essays'].count(r'@PERSON'), axis=1)
    data['location'] = data.apply(lambda x: x['corrected_essays'].count(r'@LOCATION'), axis=1)
    data['money'] = data.apply(lambda x: x['corrected_essays'].count(r'@MONEY'), axis=1)
    data['time'] = data.apply(lambda x: x['corrected_essays'].count(r'@TIME'), axis=1)
    data['date'] = data.apply(lambda x: x['corrected_essays'].count(r'@DATE'), axis=1)
    data['percent'] = data.apply(lambda x: x['corrected_essays'].count(r'@PERCENT'), axis=1)
    data['noun'] = data.apply(lambda x: x['pos'].count('NOUN'), axis=1)
    data['adj'] = data.apply(lambda x: x['pos'].count('ADJ'), axis=1)
    data['pron'] = data.apply(lambda x: x['pos'].count('PRON'), axis=1)
    data['verb'] = data.apply(lambda x: x['pos'].count('VERB'), axis=1)
    data['noun'] = data.apply(lambda x: x['pos'].count('NOUN'), axis=1)
    data['cconj'] = data.apply(lambda x: x['pos'].count('CCONJ'), axis=1)
    data['adv'] = data.apply(lambda x: x['pos'].count('ADV'), axis=1)
    data['det'] = data.apply(lambda x: x['pos'].count('DET'), axis=1)
    data['propn'] = data.apply(lambda x: x['pos'].count('PROPN'), axis=1)
    data['num'] = data.apply(lambda x: x['pos'].count('NUM'), axis=1)
    data['part'] = data.apply(lambda x: x['pos'].count('PART'), axis=1)
    data['intj'] = data.apply(lambda x: x['pos'].count('INTJ'), axis=1)

    print(f"saving to {out_file}")
    data.to_csv(out_file, sep = "\t")
    
    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default = "training_set_rel3.tsv",
                        help = "name of .tsv file containing essay data")
    parser.add_argument('--out_file', default = "training_set_rel3_x.tsv",
                        help = "name of .tsv file under which to save essay data with computed extra features")
    parser.add_argument('--glove_type', default = '6B',
                        help = "any of ['42B.300d', '840B.300d', '6B', 'twitter.27B'] "\
                        "('bin/prepare_glove.py' should have been executed for this type beforehand)")
    parser.add_argument('--dim', type = int, default = 50,
                        help = "dimension of embedding vectors "\
                        "(should be compatible with .txt files for glove type)")
    args = parser.parse_args()
    main(args)