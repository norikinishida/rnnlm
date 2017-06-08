# -*- coding: utf-8 -*-

# Script for training an RNNLM model (Elman RNN, LSTM, GRU) on a given corpus.
# Tokens in the corpus must be splitted with spaces before running this script.

import argparse
import math
import os
import sys

import numpy as np
import chainer
from chainer import cuda, serializers, optimizers
import chainer.functions as F
import pyprind

import models
import utils

def forward(model, batch_sents, train):
    # data preparation
    xs = utils.padding(batch_sents, head=True, with_mask=False)
    xs = utils.convert_ndarray_to_variable(xs, seq=True, train=train)
    # prediction
    ys = model.forward(xs, train=train)
    # loss
    ys = F.concat(ys, axis=0)
    ts = F.concat(xs, axis=0)
    ys = F.reshape(ys, (-1, model.vocab_size))
    ts = F.reshape(ts, (-1,))
    loss = F.softmax_cross_entropy(ys, ts)
    acc = F.accuracy(ys, ts, ignore_label=-1)
    return loss, acc

def evaluate(model, corpus):
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for s in pyprind.prog_bar(corpus):
        # data preparation
        batch_sents = [s]
        # forward
        loss, acc = forward(model, batch_sents, train=False)
        total_loss += loss * len(batch_sents[0])
        total_acc += acc * len(batch_sents[0])
        count += len(batch_sents[0])
    total_loss = float(cuda.to_cpu(total_loss.data)) / count
    total_acc = float(cuda.to_cpu(total_acc.data)) / count
    
    for i in xrange(5):
        # data preparation
        s = corpus.random_sample()
        xs = utils.padding([s], head=True, with_mask=False)
        xs = utils.convert_ndarray_to_variable(xs, seq=True, train=False)
        # prediction (generation)
        ys = model.generate(x_init=xs[0], train=False)
        # check
        words_ref = [corpus.ivocab[w] for w in s]
        words_gen = [words_ref[0]] + [corpus.ivocab[w[0]] for w in ys]
        utils.logger.debug("[check] <Ref.> %s" %  " ".join(words_ref))
        utils.logger.debug("[check] <Gen.> %s" %  " ".join(words_gen))

    return total_loss, total_acc

def main(gpu, path_corpus_train, path_corpus_val, path_config, path_word2vec):
    MAX_EPOCH = 10000000
    MAX_PATIENCE = 20
    EVAL = 5000
    MAX_LENGTH = 50
    
    config = utils.Config(path_config)
    basename = os.path.join(
                "rnnlm.%s.%s" % (
                os.path.basename(path_corpus_train),
                os.path.splitext(os.path.basename(path_config))[0]))
    path_snapshot = os.path.join(config.getpath("snapshot"), basename + ".model")
    path_snapshot_vectors = os.path.join(config.getpath("snapshot"), basename + ".vectors.txt")
    path_log = os.path.join(config.getpath("log"), basename + ".log")
    utils.set_logger(path_log)
    utils.logger.debug("[info] TRAINING CORPUS: %s" % path_corpus_train)
    utils.logger.debug("[info] VALIDATION CORPUS: %s" % path_corpus_val)
    utils.logger.debug("[info] CONFIG: %s" % path_config)
    utils.logger.debug("[info] PRE-TRAINED WORD EMBEDDINGS: %s" % path_word2vec)
    utils.logger.debug("[info] SNAPSHOT: %s" % path_snapshot)
    utils.logger.debug("[info] SNAPSHOT (WORD EMBEDDINGS): %s" % path_snapshot_vectors)
    utils.logger.debug("[info] LOG: %s" % path_log)
    # hyper parameters
    model_name = config.getstr("model")
    word_dim = config.getint("word_dim") 
    state_dim = config.getint("state_dim")
    grad_clip = config.getfloat("grad_clip")
    weight_decay = config.getfloat("weight_decay")
    batch_size = config.getint("batch_size")
    utils.logger.debug("[info] MODEL: %s" % model_name)
    utils.logger.debug("[info] WORD DIM: %d" % word_dim)
    utils.logger.debug("[info] STATE DIM: %d" % state_dim)
    utils.logger.debug("[info] GRADIENT CLIPPING: %f" % grad_clip)
    utils.logger.debug("[info] WEIGHT DECAY: %f" % weight_decay)
    utils.logger.debug("[info] BATCH SIZE: %d" % batch_size)
    # data preparation 
    corpus_train = utils.load_corpus(
            path_corpus_train,
            vocab=path_corpus_train + ".dictionary",
            max_length=MAX_LENGTH)
    corpus_val = utils.load_corpus(
            path_corpus_val,
            vocab=corpus_train.vocab,
            max_length=MAX_LENGTH)
    # model preparation
    if path_word2vec is not None:
        word2vec = utils.load_word2vec(path_word2vec, word_dim)
        initialW = utils.create_word_embeddings(corpus_train.vocab, word2vec, dim=word_dim, scale=0.001)
    else:
        initialW = None
    cuda.get_device(gpu).use()
    if model_name == "rnn":
        model = models.RNN(
                vocab_size=len(corpus_train.vocab),
                word_dim=word_dim,
                state_dim=state_dim,
                initialW=initialW,
                EOS_ID=corpus_train.vocab["<EOS>"])
    elif model_name == "lstm":
        model = models.LSTM(
                vocab_size=len(corpus_train.vocab),
                word_dim=word_dim,
                state_dim=state_dim,
                initialW=initialW,
                EOS_ID=corpus_train.vocab["<EOS>"])
    elif model_name == "gru":
        model = models.GRU(
                vocab_size=len(corpus_train.vocab),
                word_dim=word_dim,
                state_dim=state_dim,
                initialW=initialW,
                EOS_ID=corpus_train.vocab["<EOS>"])
    else:
        utils.logger.debug("[error] Unknown model name: %s" % model_name)
        sys.exit(-1)
    model.to_gpu(gpu)
    # training & evaluation
    opt = optimizers.SMORMS3()
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(grad_clip))
    opt.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    best_perp = np.inf
    patience = 0
    it = 0
    n_train = len(corpus_train)
    for epoch in xrange(1, MAX_EPOCH+1):
        for data_i in xrange(0, n_train, batch_size):
            if data_i + batch_size > n_train:
                break
            # data preparation
            batch_sents = corpus_train.next_batch(size=batch_size)
            # forward
            loss, acc = forward(model, batch_sents, train=True)
            # backward & update
            model.zerograds()
            loss.backward()
            loss.unchain_backward()
            opt.update()
            it += 1
            # log
            loss = float(cuda.to_cpu(loss.data))
            perp = math.exp(loss)
            acc = float(cuda.to_cpu(acc.data))
            utils.logger.debug("[training] iter=%d, epoch=%d (%d/%d=%.03f%%), perplexity=%f, accuracy=%.2f%%" \
                    % (it, epoch, data_i+batch_size, n_train,
                        float(data_i+batch_size)/n_train*100,
                        perp, acc*100))
            if it % EVAL == 0:
                # evaluation
                utils.logger.debug("[info] Evaluating on the validation sentences ...")
                loss, acc = evaluate(model, corpus_val)
                perp = math.exp(loss)
                utils.logger.debug("[validation] iter=%d, epoch=%d, perplexity=%f, accuracy=%.2f%%" \
                        % (it, epoch, perp, acc*100))
                if best_perp > perp:
                    # save
                    utils.logger.debug("[info] Best perplexity is updated: %f => %f" % (best_perp, perp))
                    best_perp = perp
                    patience = 0
                    serializers.save_npz(path_snapshot, model)
                    utils.save_word2vec(path_snapshot_vectors, utils.extract_word2vec(model, corpus_train.vocab))
                    utils.logger.debug("[info] Saved.")
                else:
                    patience += 1
                    utils.logger.debug("[info] Patience: %d (best perplexity: %f)" % (patience, best_perp))
                    if patience >= MAX_PATIENCE:
                        utils.logger.debug("[info] Patience %d is over. Training finished." % patience)
                        break

    utils.logger.debug("[info] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--corpus_train", type=str, required=True)
    parser.add_argument("--corpus_val", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--word2vec", type=str, default=None)
    args = parser.parse_args()

    gpu = args.gpu
    path_corpus_train = args.corpus_train
    path_corpus_val = args.corpus_val
    path_config = args.config
    path_word2vec = args.word2vec

    main(
        gpu=gpu,
        path_corpus_train=path_corpus_train,
        path_corpus_val=path_corpus_val,
        path_config=path_config,
        path_word2vec=path_word2vec)

