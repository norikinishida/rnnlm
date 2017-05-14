# -*- coding: utf-8 -*-

import chainer
from chainer import cuda, Variable
import chainer.functions as F
import chainer.links as L
import numpy as np


class GRU(chainer.Chain):

    def __init__(self, vocab_size, word_dim, state_dim, initialW, EOS_ID):
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.state_dim = state_dim
        # self.initialW = initialW
        self.EOS_ID = EOS_ID
        
        if initialW is not None:
            assert initialW.shape[0] == vocab_size
            assert initialW.shape[1] == word_dim
            tmp = np.random.RandomState(1234).uniform(-0.01, 0.01, (vocab_size+1, word_dim))
            tmp[0:-1, :] = initialW 
            initialW = tmp
        else:
            initialW = None
        self.vocab_size_in = self.vocab_size + 1
        self.BOS_ID = self.vocab_size_in - 1

        super(GRU, self).__init__(
            embed=L.EmbedID(self.vocab_size_in, self.word_dim,
                            ignore_label=-1, initialW=initialW),
            
            Wz_upd=L.Linear(self.word_dim, self.state_dim),
            Uz_upd=L.Linear(self.state_dim, self.state_dim, nobias=True),
            Wr_upd=L.Linear(self.word_dim, self.state_dim),
            Ur_upd=L.Linear(self.state_dim, self.state_dim, nobias=True),
            W_upd=L.Linear(self.word_dim, self.state_dim),
            U_upd=L.Linear(self.state_dim, self.state_dim, nobias=True),
           
            W_out=L.Linear(self.state_dim, self.vocab_size),
        )
        self.Uz.W.data = self.init_ortho(self.state_dim)
        self.Ur.W.data = self.init_ortho(self.state_dim)
        self.U.W.data = self.init_ortho(self.state_dim)


    def init_ortho(self, dim):
        A = np.random.randn(dim, dim)
        U, S, V = np.linalg.svd(A)
        return U.astype(np.float32)


    def forward(self, ts, train):
        N = ts[0].data.shape[0]

        s = Variable(cuda.cupy.zeros((N, self.state_dim), dtype=np.float32), volatile=not train)
        bos = Variable(cuda.cupy.full((N, 1), self.BOS_ID, dtype=np.int32), volatile=not train)
        xs = [bos] + ts[:-1]

        ys = []
        for x in xs:
            s = self.update_state(x, s, train=train)
            y = self.predict(s, train=train)
            ys.append(y)

        return ys


    def generate(self, x_init, train):
        N = x_init.data.shape[0]
        assert N == 1

        s = Variable(cuda.cupy.zeros((N, self.state_dim), dtype=np.float32), volatile=not train)
        bos = Variable(cuda.cupy.full((N, 1), self.BOS_ID, dtype=np.int32), volatile=not train)
            
        # First step. Ignore the first output.
        s = self.update_state(bos, s, train=train)
        # y = self.predict(s, train=train)
        
        x = x_init
        ys = []
        for i in xrange(50):
            s = self.update_state(x, s, train=train)
            y = self.predict(s, train=train)
            y = cuda.to_cpu(y.data) # (N=1, |V|)
            y = np.argmax(y, axis=1) # (N=1,)
            ys.append(y)

            if y[0] == self.EOS_ID:
                break
            else:
                x = Variable(cuda.cupy.asarray(y, dtype=np.int32), volatile=not train)

        return ys


    def update_state(self, x, s, train):
        v = self.embed(x)
        z = F.sigmoid(self.Wz_upd(v) + self.Uz_upd(s))
        r = F.sigmoid(self.Wr_upd(v) + self.Ur_upd(s))
        _s = F.tanh(self.W_upd(v) + self.U_upd(r * s))
        return (1.0 - z) * s + z * _s

    
    def predict(self, s, train):
        return self.W_out(s)
