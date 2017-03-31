# -*- coding: utf-8 -*-

import chainer
from chainer import cuda, Variable
import chainer.functions as F
import chainer.links as L
import numpy as np


class LSTM(chainer.Chain):

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

        super(LSTM, self).__init__(
            embed=L.EmbedID(self.vocab_size_in, self.word_dim,
                                ignore_label=-1, initialW=initialW),

            W_upd=L.Linear(self.word_dim, 4 * self.state_dim),
            U_upd=L.Linear(self.state_dim, 4 * self.state_dim, nobias=True),
            
            W_out=L.Linear(self.state_dim, self.vocab_size),
        )
        self.U_upd.W.data[self.state_dim*0:self.state_dim*1, :] = self.init_ortho(self.state_dim)
        self.U_upd.W.data[self.state_dim*1:self.state_dim*2, :] = self.init_ortho(self.state_dim)
        self.U_upd.W.data[self.state_dim*2:self.state_dim*3, :] = self.init_ortho(self.state_dim)
        self.U_upd.W.data[self.state_dim*3:self.state_dim*4, :] = self.init_ortho(self.state_dim)


    def init_ortho(self, dim):
        A = np.random.randn(dim, dim)
        U, S, V = np.linalg.svd(A)
        return U.astype(np.float32)


    def forward(self, ts=None, x_init=None, train=False):
        if ts is not None:
            ys = self.forward_with_supervision(ts, train=train)
        elif x_init is not None:
            assert x_init.data.shape[0] == 1
            ys = self.forward_without_supervision(x_init, train=train)
        else:
            print "Error: ts or x_init must not be None."
            print "ts:" % ts
            print "x_init:", x_init
            return
        return ys


    def forward_with_supervision(self, ts, train=False):
        N = ts[0].data.shape[0]

        state = {
            "h": Variable(cuda.cupy.zeros((N, self.state_dim), dtype=np.float32), volatile=not train),
            "c": Variable(cuda.cupy.zeros((N, self.state_dim), dtype=np.float32), volatile=not train),
            }
        bos = Variable(cuda.cupy.full((N, 1), self.BOS_ID, dtype=np.int32), volatile=not train)
        xs = [bos] + ts[:-1]

        ys = []
        for x in xs:
            state = self.update_state(x, state, train=train)
            y = self.predict(state["h"], train=train)
            ys.append(y)

        return ys


    def forward_without_supervision(self, x_init, train=False):
        N = x_init.data.shape[0]

        state = {
            "h": Variable(cuda.cupy.zeros((N, self.state_dim), dtype=np.float32), volatile=not train),
            "c": Variable(cuda.cupy.zeros((N, self.state_dim), dtype=np.float32), volatile=not train),
            }
        bos = Variable(cuda.cupy.full((N, 1), self.BOS_ID, dtype=np.int32), volatile=not train)
            
        # First step. Ignore the first output.
        state = self.update_state(bos, state, train=train)
        # y = self.predict(s, train=train)
        
        x = x_init
        ys = []
        for i in xrange(50):
            state = self.update_state(x, state, train=train)
            y = self.predict(state["h"], train=train)
            y = cuda.to_cpu(y.data) # (N=1, |V|)
            y = np.argmax(y, axis=1) # (N=1,)
            ys.append(y)

            if y[0] == self.EOS_ID:
                break
            else:
                x = Variable(cuda.cupy.asarray(y, dtype=np.int32), volatile=not train)

        return ys


    def update_state(self, x, state, train):
        v = self.embed(x)
        h_in = self.W_upd(v) + self.U_upd(state["h"])
        c, h = F.lstm(state["c"], h_in)
        state = {"h": h, "c": c}
        return state

    
    def predict(self, s, train):
        return self.W_out(s)
