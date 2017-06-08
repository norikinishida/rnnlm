# -*- coding: utf-8 -*-

import logging
from logging import getLogger, Formatter, StreamHandler, DEBUG
import os
import sys
import time

import numpy as np
from chainer import cuda, serializers, Variable

import models
from Config import Config
import corpus_wrapper

###############################
# global

logger = getLogger("logger")
logger.setLevel(DEBUG)

handler = StreamHandler()
handler.setLevel(DEBUG)
handler.setFormatter(Formatter(fmt="%(message)s"))
logger.addHandler(handler)

###############################
# functions 

def set_logger(filename):
    if os.path.exists(filename):
        logger.debug("[info] A file %s already exists." % filename)
        do_remove = raw_input("[info] Delete the existing log file? [y/n]: ")
        if (not do_remove.lower().startswith("y")) and (not len(do_remove) == 0):
            logger.debug("[info] Done.")
            sys.exit(0)
    logging.basicConfig(level=DEBUG, format="%(message)s", filename=filename, filemode="w")

def load_corpus(path_corpus, vocab, max_length):
    start_time = time.time()

    corpus = corpus_wrapper.CorpusWrapper(path_corpus, vocab=vocab, max_length=max_length)
    logger.debug("[info] Vocabulary size: %d" % len(corpus.vocab))

    logger.debug("[info] Checking '<EOS>' tokens ...")
    for s in corpus:
        assert s[-1] == corpus.vocab["<EOS>"]

    logger.debug("[info] Completed. %d [sec]" % (time.time() - start_time))
    return corpus

def load_word2vec(path, dim):
    word2vec = {}
    with open(path) as f:
        for line_i, line in enumerate(f):
            l = line.strip().split()
            if len(l[1:]) != dim:
                logger.debug("[info] dim %d(actual) != %d(expected), skipped line %d" % \
                        (len(l[1:]), dim, line_i+1))
                continue
            word2vec[l[0].decode("utf-8")] = np.asarray([float(x) for x in l[1:]])
    return word2vec

def create_word_embeddings(vocab, word2vec, dim, scale):
    task_vocab = vocab.keys()
    logger.debug("[info] Vocabulary size (corpus): %d" % len(task_vocab))
    word2vec_vocab = word2vec.keys()
    logger.debug("[info] Vocabulary size (pre-trained): %d" % len(word2vec_vocab))
    common_vocab = set(task_vocab) & set(word2vec_vocab)
    logger.debug("[info] Pre-trained words in the corpus: %d (%d/%d = %.2f%%)" \
        % (len(common_vocab), len(common_vocab), len(task_vocab),
            float(len(common_vocab))/len(task_vocab)*100))
    W = np.random.RandomState(1234).uniform(-scale, scale, (len(task_vocab), dim)).astype(np.float32)
    for w in common_vocab:
        W[vocab[w], :] = word2vec[w]
    return W

def make_batch(x, train, tail=True):
    N = len(x)
    max_length = max([len(x) for x in xs])
    
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
        logger.debug("[error] Unkwown model name: %s" % model_name)
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

