# -*- coding: utf-8 -*-

import cPickle as pkl
import math
import os
import sys

import numpy as np
import chainer
from chainer import cuda, serializers, optimizers, Variable
import chainer.functions as F

import config
import models
# import loss_functions
import utils


def evaluate(model, words):
    evaluator = model.copy()
    n_steps = words.size - 1
    sum_log_perp = 0.0
    state = evaluator.initialize_state(batch_size=1, train=False)
    for i in xrange(n_steps):
        # input
        x = Variable(cuda.cupy.asarray(words[i:i+1]), volatile=True)
        t = Variable(cuda.cupy.asarray(words[i+1:i+2]), volatile=True)
        # predict
        y, state = evaluator.forward(x, state, train=False)
        # loss
        loss = F.softmax_cross_entropy(y, t)
        sum_log_perp += loss.data
    perp = math.exp(float(sum_log_perp) / n_steps)
    return perp


def main(model_name, experiment):
    state_dim = config.model[model_name]["ptb"][experiment]["state_dim"]
    grad_clip = config.model[model_name]["ptb"][experiment]["grad_clip"]
    weight_decay = config.model[model_name]["ptb"][experiment]["weight_decay"]
    batch_size = config.model[model_name]["ptb"][experiment]["batch_size"]
    max_epoch = config.model[model_name]["ptb"][experiment]["max_epoch"]
    bptt_length = config.model[model_name]["ptb"][experiment]["bptt_length"]

    save_name = "%s_ptb_%s" % (model_name, experiment)

    # log
    logger = utils.MyLogger(path=os.path.join(config.log_path, save_name + ".log"))

    # dataset
    train_words, val_words, test_words, dictionary = utils.load_ptb()
    logger.write("Vocabulary size: %d" % len(dictionary.token2id))

    # model
    if model_name == "rnnlm":
        model = models.RNNLM(vocab_size=len(dictionary), state_dim=state_dim)
    elif model_name == "lstmlm":
        model = models.LSTMLM(vocab_size=len(dictionary), state_dim=state_dim)
    elif model_name == "rntnlm":
        model = models.RNTNLM(vocab_size=len(dictionary), state_dim=state_dim)
    else:
        print "Error: unknown model_name = %s" % model_name
        sys.exit(-1)
    for param in model.params():
        data = param.data
        data[:] = np.random.uniform(-0.1, 0.1, data.shape)
    cuda.get_device(0).use()
    model.to_gpu()

    # optimizer
    # opt = optimizers.Adam()
    opt = optimizers.SMORMS3()
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(grad_clip))
    opt.add_hook(chainer.optimizer.WeightDecay(weight_decay))

    # training
    # e.g., total_length = 100, batch_size = 20, jump = 5 のとき,
    #    [w_1 .. w_5 | w_2 .. w_10 | .... | w_95 .. w_100]
    # このように, ブロックが20個でき, 各ブロックは5個の連続する単語からなる.
    # 1バッチはこれら各ブロックから1つずつ単語を選択して, 合計20個(=batch_size)の単語とする.
    # したがって1エポックは5イテレーションに相当する.
    total_length = len(train_words)
    jump = total_length // batch_size
    logger.write("total_length = %d, batch_size = %d, jump = %d, bptt_length = %d" % (total_length, batch_size, jump, bptt_length))

    accum_loss = 0.0
    accum_log_perp =  cuda.cupy.zeros(())

    epoch = 1
    state = model.initialize_state(batch_size=batch_size, train=True)
    for step in xrange(jump * max_epoch):
        # input
        x = Variable(cuda.cupy.asarray([train_words[(block_i * jump + step) % total_length] for block_i in xrange(batch_size)]))
        t = Variable(cuda.cupy.asarray([train_words[(block_i * jump + step + 1) % total_length] for block_i in xrange(batch_size)]))
        # predict
        y, state= model.forward(x, state, train=True)
        # loss
        loss = F.softmax_cross_entropy(y, t)
        accum_loss += loss
        accum_log_perp += loss.data

        if (step+1) % bptt_length == 0:
            model.zerograds()
            accum_loss.backward()
            accum_loss.unchain_backward()
            opt.update()
            perp = math.exp(float(accum_log_perp) / bptt_length)
            logger.write("training: step = %d, epoch = %d, perplexity = %f" % (step+1, epoch, perp))

            accum_loss = 0.0
            accum_log_perp.fill(0)

        if (step+1) % jump == 0:
            logger.write("Evaluating ...")
            val_perp = evaluate(model, val_words)
            logger.write("validation: epoch = %d, perplexity = %f" % (epoch, val_perp))
            logger.write("Evaluating ...")
            test_perp = evaluate(model, test_words)
            logger.write("test: epoch = %d, perplexity = %f" % (epoch, test_perp))
            # save
            serializers.save_npz(os.path.join(config.snapshot_path, save_name + "_epoch_%d.model" % epoch), model)
            serializers.save_npz(os.path.join(config.snapshot_path, save_name + "_epoch_%d.opt" % epoch), opt)
            logger.write("Saved.")
            epoch += 1

    logger.write("Evaluating ...")
    test_perp = evaluate(model, test_words)
    logger.write("test: epoch = %d, perplexity = %f" % test_perp)
    serializers.save_npz(os.path.join(config.snapshot_path, save_name + "_epoch_%d.model" % epoch), model)
    serializers.save_npz(os.path.join(config.snapshot_path, save_name + "_epoch_%d.opt" % epoch), opt)
    logger.write("Saved.")

    logger.write("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: python %s <model_name> <experiment>" % sys.argv[0]
        sys.exit(-1)
    model_name = sys.argv[1]
    experiment = sys.argv[2]
    main(model_name=model_name, experiment=experiment)

