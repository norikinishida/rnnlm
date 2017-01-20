# -*- coding: utf-8 -*-

"""
入力されるコーパスは, 1行に1文ずつ記述されていること.
"""

import argparse
import os
import re

import gensim
import nltk
from nltk.tokenize import word_tokenize
from stream import *


class StartGenerator(object):

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for x in open(self.path):
            yield x


class FakeGenerator(object):

    def __init__(self, origin, f):
        self.origin = origin
        self.f = f

    def __iter__(self):
        for x in self.f(self.origin):
            yield x


class ChainGenerator(object):

    def __init__(self, *iterables):
        self.iterables = iterables

    def __iter__(self):
        for it in self.iterables:
            for x in it:
                yield x


def load_sentences(path):
    sents = StartGenerator(path)
    sents = FakeGenerator(sents,
            lambda sents_: sents_ >> map(lambda s: s.strip().decode("utf-8")))
    return sents


sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")
def tokenize(s):
    # words = []
    # for s_i in sent_detector.tokenize(s):
    #     words_i = word_tokenize(s_i)
    #     words.extend(words_i)
    # return words
    return s.split()



def replace_words_with_UNK(sents, vocab, UNK):
    identical = dict(zip(vocab, vocab))
    return FakeGenerator(sents,
            lambda sents_: sents_ >> map(lambda s: [identical.get(w, "<UNK>") for w in s]))


def write_sentences(sents, path):
    with open(path, "w") as f:
        for s in sents:
            line = " ".join(s).encode("utf-8") + "\n"
            f.write(line)


def main(path_in, path_out):
    assert os.path.exists(path_in)
    assert os.path.exists(os.path.dirname(path_out))
    prune_at = 300000
    min_count = 5

    # (0) Loading the corpus
    sents = load_sentences(path=path_in)

    # (1) Tokenizing
    sents = FakeGenerator(sents,
            lambda sents_: sents_
                >> map(lambda s: tokenize(s))
                >> filter(lambda s: len(s) != 0))

    # (2) Converting words to lower case
    sents = FakeGenerator(sents,
            lambda sents_: sents_ 
                >> map(lambda s: [w.lower() for w in s]))

    # (3) Replacing digits with '7'
    sents = FakeGenerator(sents,
            lambda sents_: sents_
                >> map(lambda s: [re.sub(r"\d", "7", w) for w in s]))

    # (4) Appending '<EOS>' tokens for each sentence
    sents = FakeGenerator(sents,
            lambda sents_: sents_
                >> map(lambda s: s + ["<EOS>"]))

    # (5.1) Constructing a temporal dictionary
    print "Constructing a temporal dictionary ..."
    dictionary = gensim.corpora.Dictionary(sents, prune_at=prune_at)
    dictionary.filter_extremes(no_below=min_count, no_above=1.0, keep_n=prune_at)
    print "Vocabulary size: %d" % len(dictionary.token2id)

    # (5.2) Replacing rare words with '<UNK>'
    sents = replace_words_with_UNK(sents, dictionary.token2id, "<UNK>")
    n_unk = 0
    n_total = 0
    for s in sents:
        for w in s:
            if w == "<UNK>":
                n_unk += 1
        n_total += len(s)
    print "# of '<UNK>' tokens: %d (%d/%d = %.2f%%)" % \
            (n_unk, n_unk, n_total, float(n_unk)/n_total*100)

    print "Writing ..."
    write_sentences(sents, path=path_out)
    print "Wrote to %s" %  path_out

    # (6.1) Reconstructing a dictionary
    dictionary = gensim.corpora.Dictionary(sents, prune_at=None)
    vocab = dictionary.token2id
    # ivocab = {w_id:w for w, w_id in vocab.items()}
    print "Vocabulary size: %d" % len(vocab)
    dictionary.save_as_text(path_out + ".dictionary")
    print "Saved the dictionary to %s" % (path_out + ".dictionary")
    
    # (6.2) Transforming words to IDs
    sents = FakeGenerator(sents,
            lambda sents_: sents_
                >> map(lambda s: [vocab[w] for w in s]))
   
    print "Writing ..."
    sents = FakeGenerator(sents,
            lambda sents_: sents_
                >> map(lambda s: [str(w) for w in s]))
    write_sentences(sents, path=path_out + ".wordids")
    print "Wrote to %s" % (path_out + ".wordids")
    print "Done."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="path to input corpus", type=str)
    parser.add_argument("-o", "--output", help="path to output corpus", type=str)
    args = parser.parse_args()

    path_in = args.input
    path_out = args.output

    main(path_in=path_in, path_out=path_out)
