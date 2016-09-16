# -*- coding: utf-8 -*-

import os

base_path = "/mnt/hdd"

data_path = os.path.join(base_path, "projects/rnnlm/data")
log_path = os.path.join(base_path, "projects/rnnlm/log")
snapshot_path = os.path.join(base_path, "projects/rnnlm/snapshot")

data= {
    "ptb": {
        "train": os.path.join(base_path, "dataset/Penn-Treebank/ptb.train.txt"),
        "val": os.path.join(base_path, "dataset/Penn-Treebank/ptb.valid.txt"),
        "test": os.path.join(base_path, "dataset/Penn-Treebank/ptb.test.txt")}}


model = {
    "rnnlm": {
        "ptb": {
            "debug": {
                "state_dim": 512,
                "batch_size": 20,
                "bptt_length": 35,
                "grad_clip": 5.0,
                "weight_decay": 4e-6,
                "max_epoch": 100
            },
            "experiment1": {
                "state_dim": 512,
                "batch_size": 20,
                "bptt_length": 35,
                "grad_clip": 5.0,
                "weight_decay": 4e-6,
                "max_epoch": 100
            },
        },
    },
    "lstmlm": {
        "ptb": {
            "debug": {
                "state_dim": 512,
                "batch_size": 20,
                "bptt_length": 35,
                "grad_clip": 5.0,
                "weight_decay": 4e-6,
                "max_epoch": 100
            },
            "experiment1": {
                "state_dim": 512,
                "batch_size": 20,
                "bptt_length": 35,
                "grad_clip": 5.0,
                "weight_decay": 4e-6,
                "max_epoch": 100
            },
        },
    },
    "rntnlm": {
        "ptb": {
            "debug": {
                "state_dim": 512,
                "batch_size": 20,
                "bptt_length": 35,
                "grad_clip": 5.0,
                "weight_decay": 4e-6,
                "max_epoch": 100,
            },
            "experiment1": {
                "state_dim": 512,
                "batch_size": 20,
                "bptt_length": 35,
                "grad_clip": 5.0,
                "weight_decay": 4e-6,
                "max_epoch": 100,
            },
        },
    }
}
