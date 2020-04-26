#!/bin/sh

cd ../bin

#python3 run_simulation.py --name baseline
#python3 run_simulation.py --name score_normalization --normalize_scores

# set normaliz_scores to true

#python3 run_simulation.py --name learning_rate --lr 1 --normalize_scores
#python3 run_simulation.py --name learning_rate --lr 0.1 --normalize_scores
#python3 run_simulation.py --name learning_rate --lr 0.01 --normalize_scores
#python3 run_simulation.py --name learning_rate --lr 0.001 --normalize_scores
#python3 run_simulation.py --name learning_rate --lr 0.0001 --normalize_scores

# set learning_rate to 0.01

#python3 run_simulation.py --name batch_size --lr 0.01 --normalize_scores --batch_size 16
#python3 run_simulation.py --name batch_size --lr 0.01 --normalize_scores --batch_size 32
#python3 run_simulation.py --name batch_size --lr 0.01 --normalize_scores --batch_size 64
#python3 run_simulation.py --name batch_size --lr 0.01 --normalize_scores --batch_size 128
#python3 run_simulation.py --name batch_size --lr 0.01 --normalize_scores --batch_size 256

# set batch_size to 256

python3 run_simulation.py --name spelling --lr 0.01 --normalize_scores --batch_size 256
python3 run_simulation.py --name spelling --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling

# set correct_spelling to true

#python3 run_simulation.py --name feat --lr 0.01 --normalize_scores --batch_size 256
#python3 run_simulation.py --name feat --lr 0.01 --normalize_scores --batch_size 256 --use_features
#python3 run_simulation.py --name feat --lr 0.01 --normalize_scores --batch_size 256 --use_features --scale_features

# set use_features to true
# set scale_features to true

#python3 run_simulation.py --name feat --lr 0.01 --normalize_scores --batch_size 256 --use_features --scale_features --dim 50
#python3 run_simulation.py --name feat --lr 0.01 --normalize_scores --batch_size 256 --use_features --scale_features --dim 300

# set dim to 300

python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 10 16
python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16
python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 16
python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128

python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 10 16 --dropout 0.2
python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --dropout 0.2
python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 16 --dropout 0.2
python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.2

python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 10 16 --dropout 0.4
python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --dropout 0.4
python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 16 --dropout 0.4
python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.4

python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 10 16 --dropout 0.6
python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --dropout 0.6
python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 16 --dropout 0.6
python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.6

python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 10 16 --dropout 0.8
python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --dropout 0.8
python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 16 --dropout 0.8
python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.8

#--dim
#--remove_stopwords
#--embedding_type
#--glove_type
#--normalize_scores
#--set_idxs
#--correct_spelling
#--features
#--use_features
#--scale_features
#--model_type
#--batch_size
#--lr
#--epochs
#--dropout
#--hidden_size
#--num_layers
#--is_bidirectional
#--use_variable_length