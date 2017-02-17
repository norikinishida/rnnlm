#!/usr/bin/env sh

BASE=/mnt/hdd
CORPUS=$BASE/projects/rnnlm/data/enwiki-latest-pages-articles.xml.corpus.preprocessed

python train.py \
    -e experiment_1 \
    -c $CORPUS \
    -g 0
