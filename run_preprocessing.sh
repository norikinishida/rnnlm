#!/usr/bin/env sh

RAW=/mnt/hdd/dataset/Wikipedia/english/corpora/enwiki-latest-pages-articles.xml.corpus
TOKENIZED=/mnt/hdd/projects/RNNLM/data/wikipedia/tmp/enwiki-latest-pages-articles.xml.corpus.tokenized
PREPROCESSED=/mnt/hdd/projects/RNNLM/data/wikipedia/enwiki-latest-pages-articles.xml.corpus.preprocessed
VOCAB=/mnt/hdd/projects/RNNLM/data/wikipedia/wikipedia.vocab.txt

# RAW=/mnt/hdd/dataset/Book-Corpus/books_large.merge.head_50000.txt
# TOKENIZED=/mnt/hdd/projects/RNNLM/data/bookcorpus/tmp/books_large.merge.head_50000.txt.tokenized
# PREPROCESSED=/mnt/hdd/projects/RNNLM/data/bookcorpus/books_large.merge.head_50000.txt.preprocessed
# VOCAB=/mnt/hdd/projects/RNNLM/data/bookcorpus/bookcorpus.vocab.txt

./preprocessing/tokenize.sh ${RAW} ${TOKENIZED}
python ./preprocessing/preprocess.py \
    --tokenized ${TOKENIZED} \
    --train ${PREPROCESSED}.train \
    --dev ${PREPROCESSED}.dev \
    --vocab ${VOCAB}
