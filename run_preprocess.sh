#!/usr/bin/env sh

BASE=/mnt/hdd
INPUT=$BASE/dataset/Wikipedia/english/corpora/enwiki-latest-articles.xml.corpus
OUTPUT=$BASE/projects/rnnlm/data/enwiki-latest-articles.xml.corpus.preprocessed

python preprocess.py \
    --input $INPUT \
    --output $OUTPUT 
