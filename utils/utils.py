# -*- coding: utf-8 -*-

from collections import Counter
import os

import numpy as np

import gensim

import config


def load_ptb(keep_n=300000):
    def load(path):
        words = open(path).read().replace("\n", "<EOS>").strip().split()
        return words
    
    print "Loading ..."
    train_words = load(path=config.data["ptb"]["train"])
    val_words = load(path=config.data["ptb"]["val"])
    test_words = load(path=config.data["ptb"]["test"])
    
    print "Filtering on word frequencies  ... (keep_n = %d)" % keep_n
    vocab = Counter(train_words + val_words + test_words)
    vocab = vocab.most_common(keep_n)
    vocab, _ = map(list, zip(*vocab))
    print "Vocabulary size: %d" % len(vocab)
    
    print "Replacing words with <UNK> ..."
    train_words = replace_words_with_UNK(train_words, vocab, "<UNK>")
    val_words = replace_words_with_UNK(val_words, vocab, "<UNK>")
    test_words = replace_words_with_UNK(test_words, vocab, "<UNK>")

    print "Constructing a dictionary ..."
    dictionary = gensim.corpora.Dictionary([train_words + val_words + test_words])
    dictionary.save_as_text(os.path.join(config.data_path, "ptb_keep_n_%d.dictionary" % keep_n))
    print "Saved the dictionary."

    print "Transforming words with word IDs ..."
    train_words = map_words_to_wordids(train_words, dictionary.token2id)
    val_words = map_words_to_wordids(val_words, dictionary.token2id)
    test_words = map_words_to_wordids(test_words, dictionary.token2id)

    print "Transforming to numpy.ndarray ..."
    train_words = np.asarray(train_words, dtype=np.int32)
    val_words = np.asarray(val_words, dtype=np.int32)
    test_words = np.asarray(test_words, dtype=np.int32)
    
    return train_words, val_words, test_words, dictionary


def replace_words_with_UNK(words, vocab, UNK):
    identical = dict(zip(vocab, vocab))
    return [identical.get(w, UNK) for w in words]


def map_words_to_wordids(words, dictionary):
    return [dictionary[w] for w in words]
