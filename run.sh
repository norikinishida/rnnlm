#!/usr/bin/env sh

RAW=/mnt/hdd/dataset/Book-Corpus/books_large.merge.head_50000.txt
CORPUS_TRAIN=/mnt/hdd/projects/rnnlm/data/bookcorpus/books_large.merge.head_50000.txt.preprocessed.train
CORPUS_VAL=/mnt/hdd/projects/rnnlm/data/bookcorpus/books_large.merge.head_50000.txt.preprocessed.val

# RAW=/mnt/hdd/dataset/Book-Corpus/books_large.merge.txt
# CORPUS_TRAIN=/mnt/hdd/projects/rnnlm/data/bookcorpus/books_large.merge.txt.preprocessed.train
# CORPUS_VAL=/mnt/hdd/projects/rnnlm/data/bookcorpus/books_large.merge.txt.preprocessed.val

##################################
# preparation

python scripts/preprocess.py \
    --raw $RAW \
    --train $CORPUS_TRAIN \
    --val $CORPUS_VAL

##################################
# training

python scripts/train.py \
    --gpu 0 \
    --corpus_train $CORPUS_TRAIN \
    --corpus_val $CORPUS_VAL \
    --config ./config/experiment_3.ini

