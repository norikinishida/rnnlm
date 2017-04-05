#!/usr/bin/env sh

RAW=/mnt/hdd/dataset/Wikipedia/english/corpora/enwiki-latest-pages-articles.xml.corpus
CORPUS=/mnt/hdd/projects/rnnlm/data/wikipedia/enwiki-latest-pages-articles.xml.corpus.preprocessed

# RAW=/mnt/hdd/dataset/Book-Corpus/books_large.merge.txt
# CORPUS=/mnt/hdd/projects/rnnlm/data/bookcorpus/books_large.merge.txt.preprocessed

TMP=./tmp.txt
python nlppreprocess/lowercase.py \
    --input $RAW \
    --output $TMP

# echo "[nlppreprocess;StanfordCoreNLP] Processing ..."
# rm tmp.properties
# touch tmp.properties
# echo "annotators = tokenize, ssplit" >> tmp.properties
# echo "ssplit.eolonly = true" >> tmp.properties
# echo "outputFormat = conll" >> tmp.properties
# echo "file = $TMP" >> tmp.properties
# java -Xmx10g edu.stanford.nlp.pipeline.StanfordCoreNLP -props tmp.properties
# python nlppreprocess/conll2lines.py \
#     --input $TMP.conll \
#     --output $TMP

python nlppreprocess/tokenizer.py \
    --input $TMP \
    --output $TMP

python nlppreprocess/replace_digits.py \
    --input $TMP \
    --output $TMP

python nlppreprocess/append_eos.py \
    --input $TMP \
    --output $TMP

python nlppreprocess/replace_rare_words.py \
    --input $TMP \
    --output $CORPUS \
    --prune_at 300000 \
    --min_count 40

python scripts/train.py \
    --gpu 0 \
    --corpus $CORPUS \
    --config ./config/template.ini
