# Final project for « Natural Language Processing » class
This is the final project for the « Natural Language Processing » class of Master IASD (Artificial Intelligence, Systems, Data), a joint PSL University Master program hosted by Paris-Dauphine, École normale supérieure, and MINES ParisTech.

## Introduction

In this project we propose and compare different neural network architectures as well as learning strategies for the automated essay scoring task.
We test our algorithms using the data from the [ASAP competition on Kaggle](https://www.kaggle.com/c/asap-aes/) sponsored by The Hewlett Foundation.

## Project structure
|Folder|Content|
|-|-|
|bin| python scripts |
|script| shell scripts to reproduce experiments |
|data| essays, features and embeddings |
|doc| report |
|src| source code |

## Requirements

You can install required packages for this project by running:
```
pip install -r requirements.txt 
```

## Preparing word embeddings

The first step when processing essays is to turn words into vectors.
Three types of word embedding are available in this project:
* Random embedding: we assign to each word a vector of random samples from a normal (Gaussian) distribution.
* Trained Word2Vec embedding: we train a Word2Vec model [1] on the sentences extracted from the training essays to capture domain-specific knowledge for this task.
* Pre-trained GloVe embedding: we load word representation given by a GloVe model [2] pre-trained on large-scale corpora such as Wikipedia.

To be able to use GloVe embedding, download pre-trained word vectors from [GloVe Repository](https://github.com/stanfordnlp/GloVe) (the lightest is `glove.6B.zip`) and extract the content of the archive in the folder `data/glove` inside the project.
Then, to parse `.txt` files into easily loadable objects go to the project folder and run:
```
cd bin
python3 prepare_glove.py --dims 50, 100, 200, 300
cd ..
```
where `--dims` flag indicates the size of the embedding vectors which should be compatible whith the `.txt` file (dimension is indicated in the name of the file).

## Extracting features

Essays contain lots of misspelled words. Corrected essays and extra features (such as part of speech indicators, ponctuation, quotations...) can be obtained by running:
```
cd bin  
python3 compute_x_features.py --data_file training_set_rel3.tsv --out_file training_set_rel3_x.tsv
cd ..
```
where `--data_file` indicates the name of a `.tsv` file saved in `data/` folder with the essays and the same formatting as for the Kaggle competition, `--out_file` is the name you want to give to the `.tsv` file with corrected essays and extra features.
As it may take quite some time to compute, we include in this repository this extra file.

## Splitting essays into train and test sets

The Kaggle competition released the scores only for the training essays. We split the data into train and test so that the test set remains unseen during the experiments.
To do this run:
```
cd bin  
python3 split_train_test.py --data_file training_set_rel3_x.tsv --train_file rel3_x_train.tsv --test_file rel3_x_test.tsv
cd ..
```
where `--data_file` indicates the name of a `.tsv` file saved in `data/` folder with the essays and `--train_file` (respectively `--test_file`) indicates the name for the outputted train (respectively test) `.tsv` file that will be saved in the same folder.

## Reproducing experimentations

To run all the experiments presented in this project:
```
cd script
chmod +x experiments.sh
./experiments.sh
cd ..
```
We compare various configuations using 5-folds cross-validation.
Results from the experiment will be saved in a table including input parameters and validation metrics and will be saved in `log/` folder in `simulation.csv` and `simulation.xlsx` files. Learning curves can be found in `log/plots` folder.

## Training and testing best model

To train the model corresponding to the best configuration found during the experiments and then test the model run:
```
cd bin
python3 run_test.py --name 'test' --train_file 'train_x.tsv' --test_file 'test_x.tsv' --dim 300 --remove_stopwords --normalize_scores --correct_spelling --use_features --scale_features --device 'cuda' --batch_size 256 --dropout 0.2 --hidden_size 300 128 --save_best_weights
cd ..
```
where `--save_best_weights` allows to save the best model weights as well as the vocab dictionary (which generates the encoding for each words in the essays) in the `checkpoint/` folder.

## References

[1] T. Mikolov, K. Chen, G. Corrado, and J. Dean, Efficient estimation of word representations in vector space, arXiv preprint, 2013.

[2] J. Pennington, R. Socher, and C. Manning, Glove: Global vectors for word representation, in Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, 2014.

