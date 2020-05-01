#!/bin/sh

cd ../bin

#python3 run_simulation.py --name baseline --device cuda
#python3 run_simulation.py --name score_normalization --normalize_scores --device cuda

# set normaliz_scores to true

#python3 run_simulation.py --name learning_rate --lr 1 --normalize_scores --device cuda
#python3 run_simulation.py --name learning_rate --lr 0.1 --normalize_scores --device cuda
#python3 run_simulation.py --name learning_rate --lr 0.01 --normalize_scores --device cuda
#python3 run_simulation.py --name learning_rate --lr 0.001 --normalize_scores --device cuda
#python3 run_simulation.py --name learning_rate --lr 0.0001 --normalize_scores --device cuda

# set learning_rate to 0.01

#python3 run_simulation.py --name batch_size --lr 0.01 --normalize_scores --batch_size 16 --device cuda
#python3 run_simulation.py --name batch_size --lr 0.01 --normalize_scores --batch_size 32 --device cuda
#python3 run_simulation.py --name batch_size --lr 0.01 --normalize_scores --batch_size 64 --device cuda
#python3 run_simulation.py --name batch_size --lr 0.01 --normalize_scores --batch_size 128 --device cuda
#python3 run_simulation.py --name batch_size --lr 0.01 --normalize_scores --batch_size 256 --device cuda

# set batch_size to 256

#python3 run_simulation.py --name spelling --lr 0.01 --normalize_scores --batch_size 256 --device cuda
#python3 run_simulation.py --name spelling --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --device cuda

# set correct_spelling to true

#python3 run_simulation.py --name feat --lr 0.01 --normalize_scores --batch_size 256 --device cuda
#python3 run_simulation.py --name feat --lr 0.01 --normalize_scores --batch_size 256 --use_features --device cuda
#python3 run_simulation.py --name feat --lr 0.01 --normalize_scores --batch_size 256 --use_features --scale_features --device cuda

# set use_features to true
# set scale_features to true

#python3 run_simulation.py --name feat --lr 0.01 --normalize_scores --batch_size 256 --use_features --scale_features --dim 50 --device cuda
#python3 run_simulation.py --name feat --lr 0.01 --normalize_scores --batch_size 256 --use_features --scale_features --dim 300 --device cuda

# set dim to 300

#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 10 16 --device cuda
#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --device cuda
#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 16 --device cuda
#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --device cuda

#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 10 16 --dropout 0.2 --device cuda
#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --dropout 0.2 --device cuda
#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 16 --dropout 0.2 --device cuda
#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.2 --device cuda

#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 10 16 --dropout 0.4 --device cuda
#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --dropout 0.4 --device cuda
#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 16 --dropout 0.4 --device cuda
#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.4 --device cuda

#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 10 16 --dropout 0.6 --device cuda
#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --dropout 0.6 --device cuda
#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 16 --dropout 0.6 --device cuda
#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.6 --device cuda

#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 10 16 --dropout 0.8 --device cuda
#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --dropout 0.8 --device cuda
#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 16 --dropout 0.8 --device cuda
#python3 run_simulation.py --name dense --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.8 --device cuda

# set hidden to 300 128 for dense
# set droppout to 0.2 for dense

#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --device cuda
#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --device cuda --use_variable_length

# set use_variable_length to true

#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 10 16 --device cuda --use_variable_length
#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --device cuda --use_variable_length
#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 128 --device cuda --use_variable_length

#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 10 16 --dropout 0.2 --device cuda --use_variable_length
#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --dropout 0.2 --device cuda --use_variable_length
#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 128 --dropout 0.2 --device cuda --use_variable_length

#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 10 16 --dropout 0.4 --device cuda --use_variable_length
#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --dropout 0.4 --device cuda --use_variable_length
#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 128 --dropout 0.4 --device cuda --use_variable_length

# set dropout to 0.2 for lstm
# set hidden to 100 16 for lstm

#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --dropout 0.2 --device cuda --use_variable_length
#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --dropout 0.2 --num_layers 2 --device cuda --use_variable_length
#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --dropout 0.2 --num_layers 3 --device cuda --use_variable_length
#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --dropout 0.2 --num_layers 4 --device cuda --use_variable_length
#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --dropout 0.2 --num_layers 5 --device cuda --use_variable_length
#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --dropout 0.2 --num_layers 6 --device cuda --use_variable_length

# set num_layers to 4 for lstm

#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --num_layers 4 --use_variable_length --device cuda
#python3 run_simulation.py --name lstm --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --num_layers 4 --is_bidirectional --use_variable_length --device cuda

# set is_bidirectional to false

#python3 run_simulation.py --name embedding --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.2 --device cuda 
#python3 run_simulation.py --name embedding --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.2 --embedding_type glove --device cuda
#python3 run_simulation.py --name embedding --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.2 --embedding_type random --device cuda

#python3 run_simulation.py --name embedding --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --num_layers 4 --use_variable_length --device cuda 
#python3 run_simulation.py --name embedding --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --num_layers 4 --use_variable_length --embedding_type glove --device cuda
#python3 run_simulation.py --name embedding --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --num_layers 4 --use_variable_length --embedding_type random --device cuda

# set embedding_type to word2vec

#python3 run_simulation.py --name sotpwords --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.2 --device cuda
#python3 run_simulation.py --name sotpwords --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.2 --device cuda --remove_stopwords

#python3 run_simulation.py --name sotpwords --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --num_layers 3 --use_variable_length --device cuda
#python3 run_simulation.py --name sotpwords --model_type lstm --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 100 16 --num_layers 3 --use_variable_length --device cuda --remove_stopwords

# set remove_stopwords to true for dense
# set remove_stopwords to ?

#python3 run_simulation.py --name sets --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.2 --device cuda --remove_stopwords
#python3 run_simulation.py --name sets --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.2 --device cuda --remove_stopwords --set_idxs 1
#python3 run_simulation.py --name sets --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.2 --device cuda --remove_stopwords --set_idxs 2
#python3 run_simulation.py --name sets --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.2 --device cuda --remove_stopwords --set_idxs 3
#python3 run_simulation.py --name sets --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.2 --device cuda --remove_stopwords --set_idxs 4
#python3 run_simulation.py --name sets --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.2 --device cuda --remove_stopwords --set_idxs 5
#python3 run_simulation.py --name sets --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.2 --device cuda --remove_stopwords --set_idxs 6
#python3 run_simulation.py --name sets --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.2 --device cuda --remove_stopwords --set_idxs 7
#python3 run_simulation.py --name sets --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.2 --device cuda --remove_stopwords --set_idxs 8

python3 run_simulation.py --name feat_only --model_type dense_feat --lr 0.01 --normalize_scores --batch_size 256 --correct_spelling --dim 300 --use_features --scale_features --hidden_size 300 128 --dropout 0.2 --remove_stopwords


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