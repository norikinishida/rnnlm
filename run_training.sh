#!/usr/bin/env sh

CORPUS=/mnt/hdd/projects/rnnlm/data/enwiki-latest-pages-articles.xml.corpus.preprocessed

python train.py \
    --gpu 0 \
    --corpus $CORPUS \
    --config ./config/experiment_1.ini
