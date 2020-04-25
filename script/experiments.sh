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

# set batch_size to 128

#python3 run_simulation.py --name embedding --lr 0.01 --normalize_scores --batch_size 128 --embedding_type word2vec
#python3 run_simulation.py --name embedding --lr 0.01 --normalize_scores --batch_size 128 --embedding_type glove
#python3 run_simulation.py --name embedding --lr 0.01 --normalize_scores --batch_size 128 --embedding_type random
#python3 run_simulation.py --name embedding --lr 0.01 --normalize_scores --batch_size 128 --embedding_type word2vec --correct_spelling
#python3 run_simulation.py --name embedding --lr 0.01 --normalize_scores --batch_size 128 --embedding_type glove --correct_spelling
#python3 run_simulation.py --name embedding --lr 0.01 --normalize_scores --batch_size 128 --embedding_type random --correct_spelling
#python3 run_simulation.py --name embedding --lr 0.01 --normalize_scores --batch_size 128 --embedding_type word2vec --correct_spelling --dim 100
#python3 run_simulation.py --name embedding --lr 0.01 --normalize_scores --batch_size 128 --embedding_type glove --correct_spelling --dim 100
#python3 run_simulation.py --name embedding --lr 0.01 --normalize_scores --batch_size 128 --embedding_type random --correct_spelling --dim 100
#python3 run_simulation.py --name embedding --lr 0.01 --normalize_scores --batch_size 128 --embedding_type word2vec --correct_spelling --dim 200
#python3 run_simulation.py --name embedding --lr 0.01 --normalize_scores --batch_size 128 --embedding_type glove --correct_spelling --dim 200
#python3 run_simulation.py --name embedding --lr 0.01 --normalize_scores --batch_size 128 --embedding_type random --correct_spelling --dim 200
#python3 run_simulation.py --name embedding --lr 0.01 --normalize_scores --batch_size 128 --embedding_type word2vec --correct_spelling --dim 300
#python3 run_simulation.py --name embedding --lr 0.01 --normalize_scores --batch_size 128 --embedding_type glove --correct_spelling --dim 300
#python3 run_simulation.py --name embedding --lr 0.01 --normalize_scores --batch_size 128 --embedding_type random --correct_spelling --dim 300

# set correct_spelling to true
# set dim to 300

python3 run_simulation.py --name features --lr 0.01 --normalize_scores --batch_size 128 --correct_spelling --dim 300
python3 run_simulation.py --name features --lr 0.01 --normalize_scores --batch_size 128 --correct_spelling --dim 300 --use_features
python3 run_simulation.py --name features --lr 0.01 --normalize_scores --batch_size 128 --correct_spelling --dim 300 --use_features --scale_features

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