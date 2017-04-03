#!/usr/bin/env sh

RAW=/mnt/hdd/dataset/Book-Corpus/books_large.merge.head_50000.txt
CORPUS=/mnt/hdd/projects/rnnlm/data/books_large.merge.head_50000.txt.preprocessed

python nlppreprocess/preprocess.py \
    --input $RAW \
    --output $CORPUS \
    --lowercase 1 \
    --replace_digits 1 \
    --append_eos 1 \
    --replace_rare 1 \
    --prune_at 300000 \
    --min_count 5

python scripts/train.py \
    --gpu 0 \
    --corpus $CORPUS \
    --config ./config/experiment_2.ini
