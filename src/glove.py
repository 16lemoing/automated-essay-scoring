# -*- coding: utf-8 -*-

import requests
import zipfile
import io
import bcolz
import numpy as np
import pickle
from tqdm import tqdm

def download_glove(glove_dir, glove_type):
    """
    Download glove embedding of chosen type and unzip it in glove dir

    Parameters
    ----------
    glove_dir : string (path to the folder where is the embedding text file, output files will be saved under the same folder)
    glove_type : string (identifier for selected glove file)
    """
    
    print(f"downloading glove {glove_type} to {glove_dir}")
    zip_file_url = f"http://nlp.stanford.edu/data/wordvecs/glove.{glove_type}.zip"
    r = requests.get(zip_file_url)
    assert r.ok, f"failed to download {zip_file_url}" 
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(glove_dir)

def preprocess_glove(glove_dir, glove_type, dim):
    """
    Parse glove embedding vectors from text file and save them for fast access

    Parameters
    ----------
    glove_dir : string (path to the folder where is the embedding text file, output files will be saved under the same folder)
    glove_type : string (identifier for selected glove file)
    dim : int (dimension of embedding vectors)
    """
    
    words = []
    idx = 0
    word2idx = {}
    vectors = []

    # Prepare embedding data file
    vectors = bcolz.carray(np.zeros(1), rootdir = glove_dir / f'{glove_type}.{dim}.dat', mode = 'w')

    # Read embeddings
    glove_txt_file = glove_dir / f'glove.{glove_type}.{dim}d.txt'
    assert glove_txt_file.exists(), f"{glove_txt_file} does not exists"
    num_lines = sum(1 for line in open(glove_txt_file, 'rb'))
    with open(glove_txt_file, 'rb') as f:
        for l in tqdm(f, total=num_lines, desc=f'reading embeddings {glove_type} of dim {dim}'):
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    # Save preprocessed embeddings
    vectors = bcolz.carray(vectors[1:].reshape((idx, dim)), rootdir = glove_dir / f'{glove_type}.{dim}.dat', mode = 'w')
    vectors.flush()
    pickle.dump(words, open(glove_dir / f'{glove_type}.{dim}_words.pkl', 'wb'))
    pickle.dump(word2idx, open(glove_dir / f'{glove_type}.{dim}_idx.pkl', 'wb'))

def get_glove(glove_dir, glove_type, dim):
  """
  Load a dictionnary of glove embedding vectors

  Parameters
  ----------
  glove_dir : string (path to the folder where is the embedding text file, output files will be saved under the same folder)
  glove_type : string (identifier for selected glove file)
  dim : int (dimension of embedding vectors)

  Returns
  -------
  glove : dictionary (key: word, value: embedding vector)
  """
  
  vectors = bcolz.open(glove_dir / f'{glove_type}.{dim}.dat')[:]
  words = pickle.load(open(glove_dir / f'{glove_type}.{dim}_words.pkl', 'rb'))
  word2idx = pickle.load(open(glove_dir / f'{glove_type}.{dim}_idx.pkl', 'rb'))
  glove = {w: vectors[word2idx[w]] for w in words}
  return glove