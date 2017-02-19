# -*- coding: utf-8 -*-

import argparse
import re

import matplotlib.pyplot as plt
import numpy as np


def extract(path):
    comp = re.compile("perplexity=([0-9]+\.[0-9]+)", re.I)
    perp_train, perp_val, perp_test = [], [], []
    for line in open(path):
        line = line.strip()
        ms = comp.findall(line)
        if len(ms) > 0:
            p = float(ms[0])
            if line.startswith("[training]"):
                perp_train.append(p)
            elif line.startswith("[validation]"):
                perp_val.append(p)
            elif line.startswith("[test]"):
                perp_test.append(p)
    return perp_train, perp_val, perp_test


def plot(xs, ys, marker=None, c="r", label="no label", title="no title", xlabel="no xlabel", ylabel="no ylabel"):
    plt.clf()
    plt.plot(xs, ys, marker=marker, c=c, label=label)
    plt.legend()
    plt.xlabel(xlabel, fontsize=20, fontname="serif")
    plt.ylabel(ylabel, fontsize=20, fontname="serif")
    plt.tick_params(labelsize=20)
    plt.title(title, fontsize=20, fontname="serif")
    plt.show()


def main(path):
    perp_train, perp_val, perp_test = extract(path)

    plot(np.arange(1, len(perp_train)+1), perp_train,
            marker=None, c="b", label="train",
            title="Training",
            xlabel="Iteration",
            ylabel="Perplexity")

    plot(np.arange(1, len(perp_val)+1), perp_val,
            marker="o", c="g", label="val",
            title="Validation",
            xlabel="Epoch",
            ylabel="Perplexity")
 
    plot(np.arange(1, len(perp_test)+1), perp_test,
            marker="o", c="r", label="test",
            title="Test",
            xlabel="Epoch",
            ylabel="Perplexity")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to log file", type=str, required=True)
    args = parser.parse_args()

    path = args.path

    main(path=path)
