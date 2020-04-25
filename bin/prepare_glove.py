# -*- coding: utf-8 -*-

import sys
sys.path.append("../src")
import argparse
from pathlib import Path

from glove import download_glove, preprocess_glove

def main(args):
    
    # Build glove directory
    glove_dir = Path("..") / "data" / "glove"
    if not glove_dir.exists():
        glove_dir.mkdir()
    
    # Download glove if needed
    if args.download:
        download_glove(glove_dir, args.glove_type)
    
    # Preprocess glove embedding
    for dim in args.dims:
        preprocess_glove(glove_dir, args.glove_type, dim)
    
    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action = 'store_true',
                        help = "to download glove from web url "\
                        "(otherwise you need to manually download archive and unzip it in 'data/glove/' dir)")
    parser.add_argument('--glove_type', default = '6B',
                        help = "any of ['42B.300d', '840B.300d', '6B', 'twitter.27B'] "\
                        "(see updated list of available types at https://github.com/stanfordnlp/GloVe)")
    parser.add_argument('--dims', type = int, nargs = '+', default = [50, 100, 200, 300],
                        help = "dimensions of embedding vectors "\
                        "(should be compatible with .txt files for glove type)")
    args = parser.parse_args()
    main(args)