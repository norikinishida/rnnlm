#!/usr/bin/env sh

# PTB
python train.py \
    -g 0 \
    -e experiment_1 \
    -c /path/to/ptb.train.txt \
    -cv /path/to/ptb.valid.txt \
    -ct /path/to/ptb.test.txt \
    -p 1
