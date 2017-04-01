#!/usr/bin/env sh

CORPUS=/mnt/hdd/projects/rnnlm/data/books_large.merge.head_50000.txt.preprocessed

python scripts/train.py \
    --gpu 0 \
    --corpus $CORPUS \
    --config ./config/experiment_2.ini
