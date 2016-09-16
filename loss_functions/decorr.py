# -*- coding: utf-8 -*-

import chainer.functions as F

import utils


def decorr(h, train):
    N = h.data.shape[0]
    D = h.data.shape[1]
    centered_h = utils.chainer_centering(h, train=train)
    C = (1.0/N) * F.matmul(centered_h, centered_h, transa=True)
    M = utils.chainer_mask4diag(D, gpu=0, train=train)
    masked_C = C * M
    return utils.chainer_frobenius_norm(masked_C)

