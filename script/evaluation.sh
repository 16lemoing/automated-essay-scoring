#!/bin/sh

cd ../bin

# Compute the importance of each word for each set as presented
# in Section 3.4.2. The output files are in ../outputs/[checkpoint_name]
# Note : you should have a trained model, its vocabulary and its scaler in
# 	../checkpoint/[checkpoint_name] before running this script.
python3 run_evaluation.py --train_file 'train_x.tsv' --checkpoint_name '000001' --model_name 'fold0_weights.pth' --vocab_name 'vocab.pkl' --remove_stopwords --normalize_scores --correct_spelling --use_features --scale_features --device 'cuda' --batch_size 16