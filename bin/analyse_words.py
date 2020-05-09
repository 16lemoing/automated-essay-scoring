import argparse
from pathlib import Path
import os
import csv
import json

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

def main(args):
    checkpoint_dir = Path('../outputs') / args.checkpoint_name
    processed_dir = checkpoint_dir / 'processed'
    processed_dir.mkdir(parents=False, exist_ok=True)
    def save_csv(l, name):
        with open(str(processed_dir/name), 'w') as f:
            csv.writer(f).writerows([[item] for item in l])

    statistics = {}

    files = glob.glob(str(checkpoint_dir/'important_words_set_*.csv'))
    files.sort()
    n_sets = len(files)

    print('found files for important words:')
    for fname in files:
        print('*',fname)

    dfs = [pd.read_csv(f, header=None, sep=";") for f in files]
    dfs = [df.sort_values(df.columns[0]).reset_index(drop=True) for df in dfs]
    for df in dfs:
        df.columns = ['Words']
    dfs = [df.set_index('Words') for df in dfs]
    for n in range(n_sets):
        df = dfs[n]
        df.loc[:,n] = True

    words = pd.concat(dfs, axis=1, join='outer')
    words = words.fillna(False)

    exactly_k_sets = [0] * n_sets
    for k in range(n_sets, 0, -1):
        words_in_k_sets = words.index[words.sum(axis=1) == k]
        save_csv(words_in_k_sets.values, f'in_{k}_sets.csv')
        exactly_k_sets[k-1] = len(words_in_k_sets.values)
        statistics[f'Words important for exactly {k} sets'] = exactly_k_sets[k-1]

    plt.figure()
    x = list(range(1, n_sets+1))
    plt.bar(x,height=exactly_k_sets)
    plt.xlabel("Number of sets")
    plt.ylabel("Number of words important for exactly $k$ sets")
    plt.grid(True)
    plt.savefig(str(processed_dir/'words_important_in_k_sets.pdf'), format='pdf')


    in_one_set = words.loc[words_in_k_sets,:]
    only_in_k = [0] * n_sets
    for k in range(n_sets):
        in_set_k = in_one_set.loc[:,k]
        only_in_set_k = in_one_set.loc[in_set_k.values, k]
        save_csv(only_in_set_k.index, f'only_in_set_{k+1}.csv')
        only_in_k[k] = len(only_in_set_k.values)
        statistics[f'Words important only in set {k+1}'] = only_in_k[k]

    plt.figure()
    x = list(range(1, n_sets+1))
    plt.bar(x, height=only_in_k)
    plt.xlabel("Set number")
    plt.ylabel("Number of words important for this set only")
    plt.grid(True)
    plt.savefig(str(processed_dir/'words_important_only_for_k.pdf'), format='pdf')

    source_based = [2,3,4,5,6]
    common_to_source_based = words.loc[:,source_based].sum(axis=1)
    common_to_source_based = words.loc[common_to_source_based == 4]
    statistics[f'Words important to sets {source_based} combined'] = len(common_to_source_based.values)
    save_csv(common_to_source_based.index.values, 'common_to_source_based.csv')

    with open(str(processed_dir/'statistics.json'), 'w') as f:
        json.dump(statistics, f, separators=(',\n',':'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_name', default='000001',
                        help='name of the checkpoint directory where the importances are saved')
    args = parser.parse_args()
    main(args)
