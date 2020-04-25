# -*- coding: utf-8 -*-

import sys
sys.path.append("../src")
import argparse
from pathlib import Path
import pandas as pd
import random

def main(args):
    
    data_dir = Path("..") / "data"
    data_file = data_dir / args.data_file
    train_file = data_dir / args.train_file
    test_file = data_dir / args.test_file

    print("loading data")
    data = pd.read_csv(data_file, sep='\t', encoding='ISO-8859-1')

    print("splitting train and test data")
    size = data.shape[0]
    idxs = list(range(size))
    random.shuffle(idxs)
    split_idx = int(args.train_split * size)
    train_idxs = idxs[:split_idx]
    test_idxs = idxs[split_idx:]
    data_train = data.iloc[train_idxs]
    data_test = data.iloc[test_idxs]
    
    if not train_file.exists() or args.overwrite:
        print(f"saving train data of size {data_train.shape[0]} to {train_file}")
        data_train.to_csv(train_file, sep="\t")
    else:
        print(f"{train_file} already exists (choose another name or append '--overwrite' argument)")
    
    if not test_file.exists() or args.overwrite:
        print(f"saving test data of size {data_test.shape[0]} to {test_file}")
        data_test.to_csv(test_file, sep="\t")
    else:
        print(f"{test_file} already exists (choose another name or append '--overwrite' argument)")
    
    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default = "training_set_rel3_x.tsv",
                        help = "name of .tsv file containing essay data")
    parser.add_argument('--train_file', default = "train_x.tsv",
                        help = "name of .tsv file under which to save train data")
    parser.add_argument('--test_file', default = "test_x.tsv",
                        help = "name of .tsv file under which to save test data")
    parser.add_argument('--train_split', type = float, default = 0.8,
                        help = "proportion of training data")
    parser.add_argument('--overwrite', action = 'store_true',
                        help = "to allow overwriting out files if they already exist")
    args = parser.parse_args()
    main(args)