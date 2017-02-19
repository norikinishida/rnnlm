#!/usr/bin/env sh

BASE=/mnt/hdd
CORPUS=$BASE/projects/rnnlm/data/enwiki-latest-pages-articles.xml.corpus.preprocessed

python train.py \
    --experiment experiment_1 \
    --corpus $CORPUS \
    --gpu 0
