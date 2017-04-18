#!/usr/bin/env sh

RAW=/mnt/hdd/dataset/Book-Corpus/books_large.merge.head_50000.txt
CORPUS_TRAIN=/mnt/hdd/projects/rnnlm/data/bookcorpus/books_large.merge.head_50000.txt.preprocessed.train
CORPUS_VAL=/mnt/hdd/projects/rnnlm/data/bookcorpus/books_large.merge.head_50000.txt.preprocessed.val

# RAW=/mnt/hdd/dataset/Book-Corpus/books_large.merge.txt
# CORPUS_TRAIN=/mnt/hdd/projects/rnnlm/data/bookcorpus/books_large.merge.txt.preprocessed.train
# CORPUS_VAL=/mnt/hdd/projects/rnnlm/data/bookcorpus/books_large.merge.txt.preprocessed.val

##################################
# preparation

TMP=./tmp.txt

python nlppreprocess/lowercase.py \
    --input $RAW \
    --output $TMP.lowercase

# echo "[nlppreprocess;StanfordCoreNLP] Processing ..."
# rm tmp.properties
# touch tmp.properties
# echo "annotators = tokenize, ssplit" >> tmp.properties
# echo "ssplit.eolonly = true" >> tmp.properties
# echo "outputFormat = conll" >> tmp.properties
# echo "file = $TMP.lowercase" >> tmp.properties
# java -Xmx10g edu.stanford.nlp.pipeline.StanfordCoreNLP -props tmp.properties
# python nlppreprocess/conll2lines.py \
#     --input $TMP.lowercase.conll \
#     --output $TMP.tokenize

python nlppreprocess/tokenizer.py \
    --input $TMP.lowercase \
    --output $TMP.tokenize

python nlppreprocess/replace_digits.py \
    --input $TMP.tokenize \
    --output $TMP.replace_digits

python nlppreprocess/append_eos.py \
    --input $TMP.replace_digits \
    --output $TMP.append_eos

python nlppreprocess/split_corpus.py \
    --all $TMP.append_eos \
    --train $TMP.train \
    --val $TMP.val \
    --size 5000

python nlppreprocess/create_dictionary.py \
    --corpus $TMP.train \
    --dict $CORPUS_TRAIN.dictionary \
    --prune_at 300000 \
    --min_count 5

python nlppreprocess/replace_rare_words.py \
    --input $TMP.train \
    --output $CORPUS_TRAIN \
    --dict $CORPUS_TRAIN.dictionary
python nlppreprocess/replace_digits.py \
    --input $TMP.val \
    --output $CORPUS_VAL \
    --dict $CORPUS_TRAIN.dictionary

##################################
# training

python scripts/train.py \
    --gpu 0 \
    --corpus_train $CORPUS_TRAIN \
    --corpus_val $CORPUS_VAL \
    --config ./config/experiment_3.ini

