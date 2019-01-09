#!/usr/bin/env sh

GPU=0
MODEL=lstmlm
CONFIG=./config/template.ini
NAME=trial1

python main.py \
    --gpu ${GPU} \
    --model ${MODEL} \
    --config ${CONFIG} \
    --name ${NAME} \
    --actiontype train

python main.py \
    --gpu ${GPU} \
    --model ${MODEL} \
    --config ${CONFIG} \
    --name ${NAME} \
    --actiontype evaluation

python main.py \
    --gpu ${GPU} \
    --model ${MODEL} \
    --config ${CONFIG} \
    --name ${NAME} \
    --actiontype dump_outputs

