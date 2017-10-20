#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import Chain, cuda
from chainer import function, functions, links, optimizer
from chainer import Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import numpy as np
import six.moves


from .util import UNKWORD, PADDING, BOS


from .util_chainer import my_variable, my_dropout, my_set_train, my_rnn_link


class BaseCNNEncoder(chainer.Chain):

    def __init__(self, emb_dim=100, window_size=3, init_emb=None,
                 hidden_dim=100, vocab_size=0, splitter=u' ', add_dim=0,
                 PAD_IDX=None):
        """
        Neural network tagger by dos (Santos and Zadrozny, ICML 2014).
        """
        assert window_size % 2 == 1, 'window_size must be odd.'
        dim = emb_dim
        hidden_dim = hidden_dim + add_dim
        self.add_dim = add_dim
        self.hidden_dim = hidden_dim
        super(BaseCNNEncoder, self).__init__(emb=L.EmbedID(vocab_size, emb_dim, ignore_label=-1),
                                             conv=L.Convolution2D(1, hidden_dim, ksize=(window_size, dim),
                                                                  stride=(1, dim), pad=(window_size // 2, 0)))
        self.splitter = splitter
        self.char_level_flag = True if self.splitter is None else False
        self.word_level_flag = not self.char_level_flag
        self.emb_dim = emb_dim
        self.window_size = window_size
        self.dim = dim
        self.PAD_IDX = PAD_IDX
        self.train = True
        # initialize embeddings
        if init_emb is not None:
            self.emb.W = init_emb

    def set_train(self, train):
        self.train = train
        my_set_train(train)

    def make_batch(self, data):
        """
        > phrase-level
        data = ['John', 'John Lenon', 'John Lenon is', 'John']
        separator = u' '
        or
        > char-level
        data = ['JOHN', 'John', 'Mike']
        separator = None
        """

        padding_size = self.window_size // 2
        padding = [self.PAD_IDX for i in six.moves.range(padding_size)]
        padding = self.xp.array(padding, dtype=self.xp.int32)
        data_num = len(data)
        ids = []
        boundaries = []
        i = 0
        i_char = 0
        ids.append(padding)

        for words in data:
            if self.char_level_flag:
                # Char-level (don't lowercase)
                ids.append(words)
                i_char += len(words)

            else:
                # Word-level
                ids.append(words)
            ids.append(padding)
            i += padding_size
            boundaries.append(i)
            i += len(words)
            boundaries.append(i)
        ids = self.xp.concatenate(ids)
        return ids, boundaries, data_num

    def char_index(self, n_length):
        i = 0
        index_list = []
        for length in n_length:
            idx = self.xp.array(range(i, i + length), self.xp.int32)
            index_list.append(idx)
            i += length
        return index_list

    def compute_vecs(self, word_ids, word_boundaries, phrase_num,
                     char_vecs=None):
        word_ids = my_variable(word_ids, volatile=not self.train)
        word_embs = self.emb(word_ids)     # total_len x dim
        word_embs_reshape = F.reshape(word_embs, (1, 1, -1, self.emb_dim))

        if self.word_level_flag and char_vecs is not None:
            # print(char_vecs.data.shape)
            # print(word_embs.data.shape)
            word_embs = F.concat([word_embs, char_vecs], axis=1)
            # print(word_embs.data.shape)
            dim = self.emb_dim + self.add_dim
            word_embs_reshape = F.reshape(word_embs, (1, 1, -1, dim))

        # 1 x 1 x total_len x dim
        # convolution
        word_emb_conv = self.conv(word_embs_reshape)
        # 1 x dim x total_len x 1
        word_emb_conv_reshape = F.reshape(word_emb_conv,
                                          (self.hidden_dim, -1))
        # max
        word_emb_conv_reshape = F.split_axis(word_emb_conv_reshape,
                                             word_boundaries, axis=1)

        embs = [F.max(word_emb_conv_word, axis=1) for i, word_emb_conv_word in
                enumerate(word_emb_conv_reshape) if i % 2 == 1]
        embs = F.concat(embs, axis=0)
        phrase_emb_conv = F.reshape(embs,
                                    (phrase_num, self.hidden_dim))
        return phrase_emb_conv

    def __call__(self, phrases, char_vecs=None):
        word_ids, boundaries, phrase_num = self.make_batch(phrases)
        vecs = self.compute_vecs(word_ids, boundaries, phrase_num,
                                 char_vecs=char_vecs)
        return vecs


class CharCNNEncoder(BaseCNNEncoder):

    def __init__(self, emb_dim=100, window_size=3, init_emb=None,
                 hidden_dim=100, vocab_size=0, PAD_IDX=None):
        """
        Neural network tagger by dos (Santos and Zadrozny, ICML 2014).
        """
        super(CharCNNEncoder, self).__init__(
            emb_dim=emb_dim, window_size=window_size, init_emb=init_emb,
            hidden_dim=hidden_dim, splitter=None, vocab_size=vocab_size,
            PAD_IDX=PAD_IDX)


class WordCNNEncoder(BaseCNNEncoder):

    def __init__(self, emb_dim=100, window_size=3, init_emb=None,
                 hidden_dim=100, add_dim=0, vocab_size=0,
                 PAD_IDX=None):
        """
        Neural network tagger by dos (Santos and Zadrozny, ICML 2014).
        """
        super(WordCNNEncoder, self).__init__(
            emb_dim=emb_dim, window_size=window_size, init_emb=init_emb,
            hidden_dim=hidden_dim, splitter=u' ', add_dim=add_dim,
            vocab_size=vocab_size, PAD_IDX=PAD_IDX)
