#!/bin/sh

cd ../bin

# Compute the data presented in Section 3.4.2 of the report.
# The output files are in ../outputs/[checkpoint_name]/processed
# Note : You should run `./evaluation.sh` before running this script.
python3 analyse_words.py --checkpoint_name '000001'