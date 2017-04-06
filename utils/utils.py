# -*- coding: utf-8 -*-

import os
import sys
import time

import numpy as np
from chainer import cuda, serializers, Variable
import gensim

import models
from Config import Config


def load_corpus(path_corpus, max_length):
    N_VAL = 5000

    start_time = time.time()
    
    # split
    print "[info] Loading the preprocessed corpus ..."
    sents = open(path_corpus)
    sents = [s.decode("utf-8").strip().split() for s in sents]

    # All sentences must be end with the "<EOS>" token
    print "[info] Checking '<EOS>' tokens ..."
    sents = [s + ["<EOS>"] if s[-1] != "<EOS>" else s for s in sents]
    
    # construct a dictionary
    if not os.path.exists(path_corpus + ".dictionary"):
        print "[info] Constructing a dictionary ..."
        dictionary = gensim.corpora.Dictionary(sents, prune_at=None)
        vocab = dictionary.token2id
    else:
        print "[info] Loading a dictionary ..."
        dictionary = gensim.corpora.Dictionary.load(path_corpus + ".dictionary")
        vocab = dictionary.token2id
    ivocab = {i:w for w,i in vocab.items()}
    print "[info] Vocabulary size: %d" % len(vocab)
    
    # transform words to IDs
    print "[info] Transforming words to IDs ..."
    sents = [[vocab[w] for w in s] for s in sents]

    # XXX: filter sentences
    print "[info] Filtering sentences with more than %d words ..." % max_length
    sents = [s for s in sents if len(s) <= max_length]

    # transform list to numpy.ndarray
    print "[info] Transforming list to numpy.ndarray"
    sents = np.asarray(sents)
    
    perm = np.random.RandomState(1234).permutation(len(sents))
    sents_train = sents[perm[0:-N_VAL]]
    sents_val = sents[perm[-N_VAL:]]
    print "[info] # of training sentences: %d" % len(sents_train)
    print "[info] # of validation sentences: %d" % len(sents_val)

    print "[info] Completed. %d [sec]" % (time.time() - start_time)
    return sents_train, sents_val, vocab, ivocab


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

