# -*- coding: utf-8 -*-

import sys

from chainer import cuda, serializers, Variable
import gensim
import numpy as np
from stream import *

import config
import models
from preprocess import StartGenerator, FakeGenerator


def load_corpus(path_corpus, max_length):
    n_val = 5000

    assert os.path.exists(path_corpus + ".wordids")
    assert os.path.exists(path_corpus + ".dictionary")
    start_time = time.time()

    print "Loading the preprocessed corpus ..."
    sents = StartGenerator(path_corpus + ".wordids")
    sents = FakeGenerator(sents, lambda x: x >> map(lambda s: s.strip().decode("utf-8").split()))
    sents = FakeGenerator(sents, lambda x: x >> map(lambda s: [int(w) for w in s]))
    sents = np.asarray(list(sents))
    perm = np.random.RandomState(1234).permutation(len(sents))
    sents_train = sents[perm[0:-n_val]]
    sents_val = sents[perm[-n_val:]]

    print "Filtering sentences with words more than %d ..." % max_length
    sents_train = np.asarray([s for s in sents_train if len(s) <= max_length])
    sents_val = np.asarray([s for s in sents_val if len(s) <= max_length])

    print "# of training sentences: %d" % len(sents_train)
    print "# of validation sentences: %d" % len(sents_val)
    
    print "Loading the dictionary ..."
    dictionary = gensim.corpora.Dictionary.load_from_text(path_corpus + ".dictionary")
    vocab = dictionary.token2id
    ivocab = {w_id:w for w, w_id in vocab.items()}
    print "Vocabulary size: %d" % len(vocab)

    print "Completed. %d [sec]" % (time.time() - start_time)
    return sents_train, sents_val, vocab, ivocab


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

