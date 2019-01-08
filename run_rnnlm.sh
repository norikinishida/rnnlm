#!/usr/bin/env sh

GPU=0
MODEL=lstmlm
CONFIG=./config/template.ini

python main.py \
    --gpu ${GPU} \
    --model ${MODEL} \
    --config ${CONFIG} \
    --actiontype train
