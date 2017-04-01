#!/usr/bin/env sh

INPUT=/mnt/hdd/dataset/Book-Corpus/books_large.merge.head_50000.txt
OUTPUT=/mnt/hdd/projects/rnnlm/data/books_large.merge.head_50000.txt.preprocessed

python nlppreprocess/preprocess.py \
    --input $INPUT \
    --output $OUTPUT \
    --prune_at 300000 \
    --min_count 5
