# -*- coding: utf-8 -*-

import os

# path_base = "/mnt/hdd"
path_base = "/home/nishida/storage/nishida/"
path_data = os.path.join(path_base, "projects/rnnlm/data")
path_log = os.path.join(path_base, "projects/rnnlm/log")
path_snapshot = os.path.join(path_base, "projects/rnnlm/snapshot")

data = {}

hyperparams = {
    "experiment_1": {
        "model": "lstm",
        "word_dim": 300,
        "state_dim": 512,
        "grad_clip": 5.0,
        "weight_decay": 4e-6,
        "batch_size": 100,
    },
}
