# -*- coding: utf-8 -*-

import gc
from itertools import chain
import re
import sys

from chainer import cuda, serializers, Variable
import gensim
import numpy as np
from stream import *

import config
import models


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


def load_corpus(path, path_val, path_test, preprocess, max_length=35):
    """
    コーパスはあらかじめtokenizingだけはされていること
    """
    prune_at=300000
    min_count = 5

    print "(0) Loading the training corpus ..."
    sents = load_sentences(path=path)
    if not None in [path_val, path_test]:
        sents_val = load_sentences(path=path_val)
        sents_test = load_sentences(path=path_test)
        n_train = len(sents)
        n_val = len(sents_val)
        n_test = len(sents_test)
    else:
        sents_val = []
        sents_test = []
    sents = chain(chain(sents, sents_va), sents_test)

    print "(1) Tokenizing ..."
    sents = FakeGenerator(sents,
            lambda sents_: sents_
                >> map(lambda s: s.split())
                >> filter(lambda s: 0 < len(s) <= max_length))

    if preprocess:
        print "(2) Converting words to lower case ..."
        sents = FakeGenerator(sents,
                lambda sents_: sents_ >> map(lambda s: [w.lower() for w in s]))

        print "(3) Appending '<EOS>' tokens for each sentence ..."
        sents = FakeGenerator(sents,
                lambda sents_: sents_ >> map(lambda s: s + ["<EOS>"]))

        print "(4) Replacing digits with '7' ..."
        sents = FakeGenerator(sents,
                lambda sents_: sents_ >> map(lambda s: [re.sub(r"\d", "7", w) for w in s]))

        print "(5) Constructing a temporal dictionary ..."
        dictionary = gensim.corpora.Dictionary(sents, prune_at=prune_at)
        dictionary.filter_extremes(no_below=min_count, no_above=1.0, keep_n=prune_at)
        print "Vocabulary size: %d" % len(dictionary.token2id)

        print "(6) Replacing rare words with '<UNK>' ..."
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
    else:
        print "Skipped (2)-(6)."

    print "(7) (Re)constructing a dictionary ..."
    dictionary = gensim.corpora.Dictionary(sents, prune_at=None)
    vocab = dictionary.token2id
    ivocab = {w_id: w for w, w_id in vocab.items()}
    print "Vocabulary size: %d" % len(vocab)

    print "(8) Transforming words to IDs ..."
    sents = np.asarray([[vocab[w] for w in s] for s in sents])
    gc.collect()
    print "# of total sentences: %d" % len(sents)

    if not None in [path_val, path_test]:
        sents_train = sents[0:n_train]
        sents_val = sents[n_train:n_train+n_val]
        sents_test = sents[n_train+n_val:n_train+n_val+n_test]
    else:
        n_eval = min(5000, len(sents) * 0.1)
        perm = np.random.RandomState(1234).permutation(len(sents))
        sents_train = sents[perm[0:-n_eval]]
        sents_val = sents[perm[-n_eval:-n_eval//2]]
        sents_test = sents[perm[-n_eval//2:]]
    print "# of training sentences: %d" % len(sents_train)
    print "# of validation sentences: %d" % len(sents_val)
    print "# of test sentences: %d" % len(sents_test)
    return sents_train, sents_val, sents_test, vocab, ivocab


def load_sentences(path):
    """
    Note: we tokenize words with space splitting.
    """
    sents = StartGenerator(path)
    sents = FakeGenerator(sents,
            lambda sents_: sents_ >> map(lambda s: s.strip().decode("utf-8")))
    return sents


def replace_words_with_UNK(sents, vocab, UNK):
    identical = dict(zip(vocab, vocab))
    return FakeGenerator(sents,
            lambda sents_: sents_ >> map(lambda s: [identical.get(w, "<UNK>") for w in s]))


def create_word_embeddings(vocab, word2vec, dim, scale):
    task_vocab = vocab.keys()
    print "Vocabulary size (corpus): %d" % len(task_vocab)
    word2vec_vocab = word2vec.keys()
    print "Vocabulary size (pre-trained): %d" % len(word2vec_vocab)
    common_vocab = set(task_vocab) & (word2vec_vocab)
    print "Pre-trained words in the corpus: %d (%d/%d = %.2f%%)" \
        % (len(common_vocab), len(common_vocab), len(task_vocab),
            float(len(common_vocab))/len(task_vocab)*100)
    W = np.random.RandomState(1234).uniform(-scale, scale, (len(task_vocab), dim)).astype(np.float32)
    for w in common_vocab:
        W[vocab[w], :] = word2vec[w]
    return W


def load_word2vec(path, dim):
    word2vec = {}
    with open(path) as f:
        for line_i, line in enumerate(f):
            l = line.strip().split()
            if len(l[1:]) != dim:
                print "dim %d(actual) != %d(expected), skipped line %d" % \
                        (len(l[1:]), dim, line_i+1)
            word2vec[l[0]] = np.asarray([float(x) for x in l[1:]])
    return word2vec


def make_batch(x, train, tail=True, xp=cuda.cupy):
    N = len(x)
    max_length = -1
    for i in xrange(N):
        l = len(x[i])
        if l > max_length:
            max_length = l

    y = np.zeros((N, max_length), dtype=np.int32)

    if tail:
        for i in xrange(N):
            l = len(x[i])
            y[i, 0:max_length-l] = -1
            y[i, max_length-l:] = x[i]
    else:
        for i in xrange(N):
            l = len(x[i])
            y[i, 0:l] = x[i]
            y[i, l:] = -1

    y = [Variable(xp.asarray(y[:,j]), volatile=not train)
            for j in xrange(y.shape[1])]

    return y


def load_model(path, experiment, vocab):
    model_name = config.hyperparams[experiment]["model"]
    word_dim = config.hyperparams[experiment]["word_dim"]
    state_dim = config.hyperparams[experiment]["state_dim"]

    if model_name == "rnn":
        model = models.RNN(
                vocab_size=len(vocab),
                word_dim=word_dim,
                state_dim=state_dim,
                word_embeddings=None,
                EOS_ID=vocab["<EOS>"])
    elif model_name == "lstm":
        model = models.LSTM(
                vocab_size=len(vocab),
                word_dim=word_dim,
                state_dim=state_dim,
                word_embeddings=None,
                EOS_ID=vocab["<EOS>"])
    elif model_name == "gru":
        model = models.GRU(
                vocab_size=len(vocab),
                word_dim=word_dim,
                state_dim=state_dim,
                word_embeddings=None,
                EOS_ID=vocab["<EOS>"])
    else:
        print "Unkwown model name: %s" % model_name
        sys.exit(-1)
    serializers.load_npz(path, model)
    return model


def extract_word2vec(model, vocab):
    word2vec = {}
    for w in vocab.keys():
        word2vec[w] = cuda.to_cpu(model.embed.W.data[vocab[w]])
    return word2vec


def save_word2vec(path, word2vec):
    with open(path, "w") as f:
        for w, v in word2vec.items():
            line = " ".join([w] + [str(v_i) for v_i in v]).encode("utf-8") + "\n"
            f.write(line)

