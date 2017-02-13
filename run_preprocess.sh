#!/usr/bin/env sh

INPUT=ptb.txt
OUTPUT=./ptb.txt.preprocessed

python preprocess.py \
    -i $INPUT \
    -o $OUTPUT 
