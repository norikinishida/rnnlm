import argparse
import math
import os
import time

import numpy as np
import chainer
from chainer import cuda, serializers
import chainer.functions as F
import pyprind

import utils

import dataloader
import models

###############################

MAX_EPOCH = 10000000
MAX_PATIENCE = 20
ITERS_PER_VALID = 1000
MAX_LENGTH = 30

###############################
# Saving word vectors

def save_word_vectors(path, model, vocab):

    eps = 1e-6
    def normalize(v):
        return v / (np.linalg.norm(v) + eps)

    with open(path, "w") as f, \
         open(path + ".normalized", "w") as f_norm:
        for word in vocab.keys():
            word_id = vocab[word]
            word_vector = cuda.to_cpu(model.embed.W.data[word_id])
            word_vector_normalized = normalize(word_vector)
            word_vector = [str(x) for x in word_vector]
            word_vector_normalized = [str(x) for x in word_vector_normalized]
            f.write("%s\n" % " ".join([word] + word_vector))
            f_norm.write("%s\n" % " ".join([word] + word_vector_normalized))

###############################
# Evaluation

def evaluate(model, datapool):
    """
    :type model: chainer.Chain
    :type datapool: DataPool
    :rtype: float, float
    """
    batch_size = 100

    total_loss_data = 0.0
    total_acc_data = 0.0
    n_instances = 0

    sentences = []
    for s, in pyprind.prog_bar(datapool):
        # Mini-batch preparation
        sentences.append(s)

        if len(sentences) == batch_size:
            # Forward
            seq_batch_word = model.preprocess(sentences) # max_length * (batch_size,)
            seq_batch_logits = model.forward(seq_batch_word) # max_length * (batch_size, vocab_size)

            # Loss
            batch_logits = F.concat(seq_batch_logits, axis=0) # (max_length, batch_size, vocab_size)
            batch_words = F.concat(seq_batch_word, axis=0) # (max_length, batch_size)
            batch_logits = F.reshape(batch_logits, (-1, model.vocab_size)) # (max_length * batch_size, vocab_size)
            batch_words = F.reshape(batch_words, (-1,)) # (max_length * batch_size,)
            loss = F.softmax_cross_entropy(batch_logits, batch_words)
            acc = F.accuracy(batch_logits, batch_words, ignore_label=-1)

            # Accumulate
            loss_data = float(cuda.to_cpu(loss.data))
            acc_data = float(cuda.to_cpu(acc.data))
            total_loss_data += loss_data * len(sentences)
            total_acc_data += acc_data * len(sentences)
            n_instances += len(sentences)

            # Init.
            sentences = []

    if len(sentences) != 0:
        # Forward
        seq_batch_word = model.preprocess(sentences) # max_length * (batch_size,)
        seq_batch_logits = model.forward(seq_batch_word) # max_length * (batch_size, vocab_size)

        # Loss
        batch_logits = F.concat(seq_batch_logits, axis=0) # (max_length, batch_size, vocab_size)
        batch_words = F.concat(seq_batch_word, axis=0) # (max_length, batch_size)
        batch_logits = F.reshape(batch_logits, (-1, model.vocab_size)) # (max_length * batch_size, vocab_size)
        batch_words = F.reshape(batch_words, (-1,)) # (max_length * batch_size,)
        loss = F.softmax_cross_entropy(batch_logits, batch_words)
        acc = F.accuracy(batch_logits, batch_words, ignore_label=-1)

        # Accumulate
        loss_data = float(cuda.to_cpu(loss.data))
        acc_data = float(cuda.to_cpu(acc.data))
        total_loss_data += loss_data * len(sentences)
        total_acc_data += acc_data * len(sentences)
        n_instances += len(sentences)

    total_loss_data /= n_instances
    total_acc_data /= n_instances
    return total_loss_data, total_acc_data

###############################
# Analysis

def dump_outputs(path, model, datapool):
    """
    :type path: str
    :type model: chainer.Chain
    :type datapool: DataPool
    :rtype: None
    """
    SEED_STEPS = 3

    with open(path, "w") as f:
        i = 0
        for s, in pyprind.prog_bar(datapool):
            if len(s) <= SEED_STEPS:
                f.write("[%d] [original]  %s\n" % (i, " ".join(s)))
                f.write("[%d] Skipped.\n" % i)
                f.write("#####################\n")
                continue

            initial_words = s[:SEED_STEPS]
            generated_words = model.generate_sentence(initial_words) # list of str

            f.write("[%d] [corpus]    %s\n" % (i, " ".join(s)))
            f.write("[%d] [seed]      %s\n" % (i, " ".join(initial_words)))
            f.write("[%d] [generated] %s\n" % (i, " ".join(generated_words)))
            f.write("#####################\n")
            i += 1

def interaction(model, vocab):
    identical = {w:w for w in vocab.keys()}

    while True:
        query = input("Initial words: ")
        query = query.lower()

        if query== "q":
            break

        initial_words = [identical.get(w, "<UNK>") for w in query.split()]
        generated_words = model.generate_sentence(initial_words) # list of str
        print("[beginning] %s" % " ".join(initial_words))
        print("[generated] %s" % " ".join(generated_words))

###############################
# Main

def main(args):

    ######################
    # Arguments
    gpu = args.gpu
    model_name = args.model
    path_config = args.config
    trial_name = args.name
    actiontype = args.actiontype

    # Check
    assert actiontype in ["train", "evaluation", "dump_outputs", "interaction"]

    if trial_name is None:
        trial_name = utils.get_current_time()

    ######################
    # Path setting
    config = utils.Config(path_config)

    basename = "%s.%s.%s" % (model_name,
                             utils.get_basename_without_ext(path_config),
                             trial_name)

    path_snapshot = os.path.join(config.getpath("snapshot"), basename + ".model")
    path_snapshot_vectors = os.path.join(config.getpath("snapshot"), basename + ".vectors.txt")
    path_log = os.path.join(config.getpath("log"), basename + ".log")
    path_eval = os.path.join(config.getpath("evaluation"), basename + ".eval")
    path_anal = os.path.join(config.getpath("analysis"), basename)

    if actiontype == "train":
        utils.set_logger(path_log)
    elif actiontype == "evaluation":
        utils.set_logger(path_eval)

    ######################
    # Log so far
    utils.writelog("args", "gpu=%d" % gpu)
    utils.writelog("args", "model_name=%s" % model_name)
    utils.writelog("args", "path_config=%s" % path_config)
    utils.writelog("args", "trial_name=%s" % trial_name)
    utils.writelog("args", "actiontype=%s" % actiontype)

    utils.writelog("path", "path_snapshot=%s" % path_snapshot)
    utils.writelog("path", "path_snapshot_vectors=%s" % path_snapshot_vectors)
    utils.writelog("path", "path_log=%s" % path_log)
    utils.writelog("path", "path_eval=%s" % path_eval)
    utils.writelog("path", "path_anal=%s" % path_anal)

    ######################
    # Data preparation
    begin_time = time.time()

    train_datapool = dataloader.read_corpus(config.getpath("train_corpus"))
    dev_datapool = dataloader.read_corpus(config.getpath("dev_corpus"))
    vocab = utils.read_vocab(config.getpath("vocab"))

    end_time = time.time()
    utils.writelog("corpus", "Loaded the data. %f [sec.]" % (end_time - begin_time))
    utils.writelog("corpus", "# of training sentences=%d" % len(train_datapool))
    utils.writelog("corpus", "# of development sentences=%d" % len(dev_datapool))

    ######################
    # Hyper parameters
    word_dim = config.getint("word_dim")
    state_dim = config.getint("state_dim")
    grad_clip = config.getfloat("grad_clip")
    weight_decay = config.getfloat("weight_decay")
    batch_size = config.getint("batch_size")
    optimizer_name = config.getstr("optimizer_name")

    utils.writelog("hyperparams", "word_dim=%d" % word_dim)
    utils.writelog("hyperparams", "state_dim=%d" % state_dim)
    utils.writelog("hyperparams", "grad_clip=%f" % grad_clip)
    utils.writelog("hyperparams", "weight_decay=%f" % weight_decay)
    utils.writelog("hyperparams", "batch_size=%d" % batch_size)
    utils.writelog("hyperparams", "optimizer_name=%s" % optimizer_name)

    ######################
    # Model preparation
    cuda.get_device(gpu).use()

    # Model creation
    if model_name == "rnnlm":
        model = models.RNNLM(vocab=vocab,
                             word_dim=word_dim,
                             state_dim=state_dim,
                             BOS="<BOS>",
                             EOS="<EOS>")
    elif model_name == "lstmlm":
        model = models.LSTMLM(vocab=vocab,
                              word_dim=word_dim,
                              state_dim=state_dim,
                              BOS="<BOS>",
                              EOS="<EOS>")
    else:
        raise ValueError("Unknown model_name=%s" % model_name)
    utils.writelog("model", "Initialized the model ``%s''" % model_name)

    # Model loading
    if actiontype != "train":
        serializers.load_npz(path_snapshot, model)
        utils.writelog("model", "Loaded trained parameters from %s" % path_snapshot)

    model.to_gpu(gpu)

    ######################
    # Training, Evaluation, Analysis
    if actiontype == "train":
        with chainer.using_config("train", True):
            # Optimizer preparation
            opt = utils.get_optimizer(optimizer_name)
            opt.setup(model)
            if weight_decay > 0.0:
                opt.add_hook(chainer.optimizer.WeightDecay(weight_decay))
            if grad_clip > 0.0:
                opt.add_hook(chainer.optimizer.GradientClipping(grad_clip))

            n_train = len(train_datapool)
            it = 0
            bestscore_holder = utils.BestScoreHolder(scale=1.0, higher_is_better=False)
            bestscore_holder.init()

            # Initial validation
            with chainer.using_config("train", False), chainer.no_backprop_mode():
                loss_data, acc_data = evaluate(model, dev_datapool)
                perp = math.exp(loss_data)
                utils.writelog("dev", "iter=0, epoch=0, loss=%f, perplexity=%f, accuracy=%.02f%%" % \
                        (loss_data, perp, acc_data * 100.0))

            for epoch in range(1, MAX_EPOCH+1):
                for inst_i in range(0, n_train, batch_size):
                    if inst_i + batch_size > n_train:
                        break

                    # Mini-batch preparation
                    sentences, = train_datapool.get_instances(batch_size=batch_size) # list of list of str
                    # Forward
                    seq_batch_word = model.preprocess(sentences) # max_length * (batch_size,)
                    seq_batch_logits = model.forward(seq_batch_word) # max_length * (batch_size, vocab_size)
                    # Loss
                    batch_logits = F.concat(seq_batch_logits, axis=0) # (max_length, batch_size, vocab_size)
                    batch_words = F.concat(seq_batch_word, axis=0) # (max_length, batch_size)
                    batch_logits = F.reshape(batch_logits, (-1, model.vocab_size)) # (max_length * batch_size, vocab_size)
                    batch_words = F.reshape(batch_words, (-1,)) # (max_length * batch_size,)
                    loss = F.softmax_cross_entropy(batch_logits, batch_words)
                    acc = F.accuracy(batch_logits, batch_words, ignore_label=-1)

                    # Backward & Update
                    model.zerograds()
                    loss.backward()
                    loss.unchain_backward()
                    opt.update()
                    it += 1

                    # Write log
                    loss_data = float(cuda.to_cpu(loss.data))
                    perp = math.exp(loss_data)
                    acc_data = float(cuda.to_cpu(acc.data))
                    utils.writelog("training", "iter=%d, epoch=%d (%d/%d=%.03f%%), loss=%f, perplexity=%f, accuracy=%.02f%%" % \
                            (it, epoch, inst_i+batch_size, n_train,
                            float(inst_i+batch_size)/n_train*100.0,
                            loss_data, perp, acc_data * 100.0))

                    if it % ITERS_PER_VALID != 0:
                        continue

                    # Validation
                    with chainer.using_config("train", False), chainer.no_backprop_mode():
                        loss_data, acc_data = evaluate(model, dev_datapool)
                        perp = math.exp(loss_data)
                        utils.writelog("dev", "iter=%d, epoch=%d, loss=%f, perplexity=%f, accuracy=%.02f%%" % \
                                (it, epoch, loss_data, perp, acc_data * 100.0))

                    # Saving
                    did_update = bestscore_holder.compare_scores(perp, it)
                    if did_update:
                        serializers.save_npz(path_snapshot, model)
                        save_word_vectors(path_snapshot_vectors, model, vocab)

                    # Finished?
                    if bestscore_holder.ask_finishing(max_patience=MAX_PATIENCE):
                        utils.writelog("info", "Patience %d is over. Training finished successfully." % bestscore_holder.patience)
                        utils.writelog("info", "Done.")
                        return

    elif actiontype == "evaluation":
        with chainer.using_config("train", False), chainer.no_backprop_mode():
            loss_data, acc_data = evaluate(model, dev_datapool)
            perp = math.exp(loss_data)
            utils.writelog("dev", "loss=%f, perplexity=%f, accuracy=%.02f%% " % \
                    (loss_data, perp, acc_data * 100.0))

    elif actiontype == "dump_outputs":
        with chainer.using_config("train", False), chainer.no_backprop_mode():
            dump_outputs(path_anal + ".outputs", model, dev_datapool)

    elif actiontype == "interaction":
        with chainer.using_config("train", False), chainer.no_backprop_mode():
            interaction(model, vocab)

    utils.writelog("info", "Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--actiontype", type=str, required=True)
    args = parser.parse_args()
    main(args)
