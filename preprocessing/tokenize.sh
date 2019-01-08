#!/usr/bin/env sh

java edu.stanford.nlp.process.PTBTokenizer \
    --lowerCase \
    --preserveLines \
    < $1 > $2

