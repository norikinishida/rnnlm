# -*- coding: utf-8 -*-

import chainer.functions as F
from chainer import cuda, Variable
import numpy as np


def chainer_mask4diag(dim, gpu, train):
    mask_data = np.eye(dim, dtype=np.float32)
    mask_data[mask_data == 1.0] = 2.0
    mask_data[mask_data == 0.0] = 1.0
    mask_data[mask_data == 2.0] = 0.0
    mask = Variable(cuda.to_gpu(mask_data, device=gpu), volatile=not train)
    return mask


def chainer_mean(M):
    """
    M.data.shape = (N, D)
    output: shape = ()
    """
    with cuda.get_device(M.data):
        Z = M.data.shape[0] * M.data.shape[1]
        mean = F.sum(M) * (1.0 / Z)
        return mean


def chainer_batch_normalize(M, train):
    """
    M.data.shape = (N,D)
    output: shape = (N,D)
    """
    with cuda.get_device(M.data):
        N = M.data.shape[0]
        D = M.data.shape[1]
        # norms = chainer_diag(F.matmul(M, M, transb=True), train=train)
        # norms = F.basic_math.pow(norms, 0.5)
        # norms = F.reshape(norms, (N,1))
        # ones = Variable(cuda.cupy.ones((1,D), dtype=np.float32), volatile=not train)
        # norms = F.matmul(norms, ones)
        norms = F.basic_math.pow(F.batch_l2_norm_squared(M), 0.5)
        norms = F.broadcast_to(F.reshape(norms, (-1,1)), M.data.shape)
        return M / norms


def chainer_frobenius_norm(M):
    """
    M.data.shape = (# of rows, # of columns)
    output: shape = ()
    """
    with cuda.get_device(M.data):
        # N_rows = M.data.shape[0]
        # N_cols = M.data.shape[1]
        # v = F.reshape(M, (1, N_rows*N_cols))
        v = F.reshape(M, (1,-1))
        norm = F.basic_math.pow(F.matmul(v, v, transb=True), 0.5)
        return F.reshape(norm, ())


def chainer_squared_error(v1, v2):
    """
    {v1,v2}.data.shape = (1, D)
    output: shape = ()
    """
    with cuda.get_device(v1.data):
        v = v1 - v2
        res = F.matmul(v, v, transb=True)
        return F.reshape(res, ())


def chainer_frobenius_squared_error(M1, M2):
    """
    M1.data.shape = M2.data.shape = (# of rows, # of columns)
    output: shape = ()
    """
    with cuda.get_device(M1.data):
        v1 = F.reshape(M1, (1,-1))
        v2 = F.reshape(M2, (1,-1))
        res = chainer_squared_error(v1, v2)
        return res


def chainer_diag(M, train):
    """
    M.data.shape = (# of rows, # of columns)
    output: shape = (1, min(# of rows, # of columns))
    """
    with cuda.get_device(M.data):
        N_rows = M.data.shape[0]
        index_data = cuda.cupy.arange(0, N_rows).astype(np.int32)
        index = Variable(cuda.to_gpu(index_data), volatile=not train)
        return F.reshape(F.select_item(M, index), (1, -1))


def chainer_innerprod(v1, v2):
    """
    v1.data.shape = v2.data.shape = (1, D)
    output: shape = ()
    """
    with cuda.get_device(v1.data):
        res = F.matmul(v1, v2, transb=True)
        res = F.reshape(res, ())
        return res


def chainer_trace(M, train):
    """
    M.data.shape = (# of rows, # of columns)
    output: shape = ()
    """
    with cuda.get_device(M.data):
        diag = chainer_diag(M, train)
        res = F.sum(diag)
        return res


def chainer_centering(M, train):
    with cuda.get_device(M.data):
        N = M.data.shape[0]
        ones = Variable(cuda.cupy.ones((N,N), dtype=np.float32), volatile=not train)
        return M - (1.0/N) * F.matmul(ones, M)


def chainer_shuffle(M, train):
    with cuda.get_device(M.data):
        M_data = cuda.to_cpu(M.data)
        perm = np.random.permutation(M_data.shape[0])
        return Variable(cuda.to_gpu(M_data[perm]), volatile=not train)


def chainer_negative_sampling(M, K, train):
    neg_list = []
    for k in xrange(K):
        shuffled_M = chainer_shuffle(M, train=train)
        neg_list.append(shuffled_M)
    return neg_list

