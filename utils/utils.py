# -*- coding: utf-8 -*-

import os
import sys
import time

import numpy as np
from chainer import cuda, serializers, Variable

import models
from Config import Config
from corpus_wrapper import CorpusWrapper


def load_corpus(path_corpus, vocab, max_length):
    if vocab is None:
        if not os.path.exists(path_corpus + ".dictionary"):
            print "[error] You should run nlppreprocess/create_dictionary.py before this script."
            sys.exit(-1)

    start_time = time.time()

    corpus = CorpusWrapper(path_corpus, vocab=vocab, max_length=max_length)
    print "[info] Vocabulary size: %d" % len(corpus.vocab)

    print "[info] Checking '<EOS>' tokens ..."
    for s in corpus:
        assert s[-1] == corpus.vocab["<EOS>"]

    print "[info] Completed. %d [sec]" % (time.time() - start_time)
    return corpus


def load_word2vec(path, dim):
    word2vec = {}
    with open(path) as f:
        for line_i, line in enumerate(f):
            l = line.strip().split()
            if len(l[1:]) != dim:
                print "[info] dim %d(actual) != %d(expected), skipped line %d" % \
                        (len(l[1:]), dim, line_i+1)
                continue
            word2vec[l[0].decode("utf-8")] = np.asarray([float(x) for x in l[1:]])
    return word2vec


def create_word_embeddings(vocab, word2vec, dim, scale):
    task_vocab = vocab.keys()
    print "[info] Vocabulary size (corpus): %d" % len(task_vocab)
    word2vec_vocab = word2vec.keys()
    print "[info] Vocabulary size (pre-trained): %d" % len(word2vec_vocab)
    common_vocab = set(task_vocab) & set(word2vec_vocab)
    print "[info] Pre-trained words in the corpus: %d (%d/%d = %.2f%%)" \
        % (len(common_vocab), len(common_vocab), len(task_vocab),
            float(len(common_vocab))/len(task_vocab)*100)
    W = np.random.RandomState(1234).uniform(-scale, scale, (len(task_vocab), dim)).astype(np.float32)
    for w in common_vocab:
        W[vocab[w], :] = word2vec[w]
    return W


def make_batch(x, train, tail=True):
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

    y = [Variable(cuda.cupy.asarray(y[:,j]), volatile=not train)
            for j in xrange(y.shape[1])]

    return y


def load_model(path_model, path_config, vocab):
    config = Config(path_config)
    model_name = config.getstr("model")
    word_dim = config.getint("word_dim")
    state_dim = config.getint("state_dim")

    if model_name == "rnn":
        model = models.RNN(
                vocab_size=len(vocab),
                word_dim=word_dim,
                state_dim=state_dim,
                initialW=None,
                EOS_ID=vocab["<EOS>"])
    elif model_name == "lstm":
        model = models.LSTM(
                vocab_size=len(vocab),
                word_dim=word_dim,
                state_dim=state_dim,
                initialW=None,
                EOS_ID=vocab["<EOS>"])
    elif model_name == "gru":
        model = models.GRU(
                vocab_size=len(vocab),
                word_dim=word_dim,
                state_dim=state_dim,
                initialW=None,
                EOS_ID=vocab["<EOS>"])
    else:
        print "[error] Unkwown model name: %s" % model_name
        sys.exit(-1)
    serializers.load_npz(path_model, model)
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

