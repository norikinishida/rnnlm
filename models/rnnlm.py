import chainer
from chainer import cuda, Variable
import chainer.functions as F
import chainer.links as L
import numpy as np

import utils

class RNNLM(chainer.Chain):

    def __init__(self, vocab, word_dim, state_dim, BOS, EOS):
        """
        :type vocab: {str: int}
        :type word_dim: int
        :type state_dim: int
        :type BOS: str
        :type EOS: str
        """
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.ivocab = {i:w for w,i in self.vocab.items()}
        self.word_dim = word_dim
        self.state_dim = state_dim
        self.BOS = BOS
        self.EOS = EOS
        self.BOS_ID = self.vocab[self.BOS]
        self.EOS_ID = self.vocab[self.EOS]

        links = {}
        # Word embedding
        links["embed"] = L.EmbedID(self.vocab_size, self.word_dim, ignore_label=-1)
        # RNN
        links["W_upd"] = L.Linear(self.word_dim, self.state_dim)
        links["U_upd"] = L.Linear(self.state_dim, self.state_dim, nobias=True)
        # Output
        links["W_out"] = L.Linear(self.state_dim, self.vocab_size)
        super(RNNLM, self).__init__(**links)

    def preprocess(self, batch_words):
        """
        :type batch_words: list of list of str
        :rtype: list of Variable(shape=(batch_size,), dtype=np.int32)
        """
        batch_words = [[self.vocab[w] for w in words] for words in batch_words] # list of list of int
        batch_words = utils.padding(batch_words, head=True, with_mask=False) # (batch_size, max_length)
        seq_batch_word = utils.convert_ndarray_to_variable(batch_words, seq=True) # max_length * (batch_size,)
        return seq_batch_word

    def forward(self, seq_batch_word):
        """
        :type seq_batch_word: list of Variable(shape=(batch_size,), dtype=np.int32)
        :rtype: list of Variable(batch_size, vocab_size)
        """
        batch_size = seq_batch_word[0].shape[0]

        # Initialization
        hidden_state, bos = self.make_initial_variables(batch_size=batch_size) # (batch_size, state_dim), (batch_size, 1)

        # Recurret computation
        seq_batch_logits = [] # list of Variable(shape=(batch_size, vocab_size), dtype=np.float32)
        seq_batch_inputword = [bos] + seq_batch_word[:-1] # max_length * (batch_size,)
        for batch_inputword in seq_batch_inputword:
            hidden_state = self.update_state(batch_inputword, hidden_state) # (batch_size, state_dim)
            batch_logits = self.predict_next_word(hidden_state) # (batch_size, vocab_size)
            seq_batch_logits.append(batch_logits)

        return seq_batch_logits

    def generate_sentence(self, initial_words):
        """
        :type initial_words: list of str
        :rtype: list of str
        """
        assert len(initial_words) > 0
        initial_words = self.preprocess([initial_words]) # 2 * (1,)

        # Initialization
        hidden_state, bos = self.make_initial_variables(batch_size=1) # (1, state_dim), (1, 1)

        seq_outputword = [] # list of str

        inputword = bos # (1,)
        for step in range(len(initial_words)):
            # Predict the next word
            hidden_state = self.update_state(inputword, hidden_state) # (1, state_dim)
            # NOTE that, in this loop, we are interested in only updating hidden states and ignore the predicted words.
            outputword = initial_words[step] # NOTE Variable(shape=(1,), dtype=np.int32)
            # Recode
            seq_outputword.append(int(cuda.to_cpu(outputword.data)))
            # Prepare for the next step
            inputword = outputword # (1,)

        for _ in range(50):
            # Predict the next word
            hidden_state = self.update_state(inputword, hidden_state) # (1, state_dim)
            logits = self.predict_next_word(hidden_state) # (1, vocab_size)
            logits = cuda.to_cpu(logits.data) # (1, vocab_size)
            outputword = np.argmax(logits, axis=1) # NOTE numpy.ndarray(shape=(1,), dtype=np.int32)
            # Recode
            seq_outputword.append(int(outputword))
            # Prepare for the next step
            inputword = utils.convert_ndarray_to_variable(outputword, seq=False) # (1,)

            if outputword[0] == self.EOS_ID:
                break

        seq_outputword = [self.ivocab[w] for w in seq_outputword]
        return seq_outputword

    def make_initial_variables(self, batch_size):
        """
        :type batch_size: int
        :rtype: Variable(shape=(batch_size, state_dim), dtype=np.float32), Variable(shape=(batch_size,1), dtype=np.int32)
        """
        hidden_state = Variable(cuda.cupy.zeros((batch_size, self.state_dim), dtype=np.float32)) # (batch_size, state_dim)
        bos = Variable(cuda.cupy.full((batch_size,1), self.BOS_ID, dtype=np.int32)) # (batch_size, 1)
        return hidden_state, bos

    def update_state(self, inputwords, hidden_state):
        """
        :type inputwords: Variable(shape=(batch_size,), dtype=np.int32)
        :type hidden_state: Variable(shape=(batch_size, state_dim), dtype=np.float32)
        :rtype: Variable(shape=(batch_size, state_dim), dtype=np.float32)
        """
        word_vectors = self.embed(inputwords) # (batch_size, word_dim)
        hidden_state = F.tanh(self.W_upd(word_vectors) + self.U_upd(hidden_state)) # (batch_size, state_dim)
        return hidden_state

    def predict_next_word(self, hidden_state):
        """
        :type hidden_state: Variable(shape=(batch_size, state_dim), dtype=np.float32)
        :rtype: Variable(shape=(batch_size, vocab_size), dtype=np.float32)
        """
        logits = self.W_out(hidden_state)
        return logits
