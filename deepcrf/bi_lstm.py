#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
random.seed(1234)
np.random.seed(1234)

import chainer
from chainer import Chain, cuda
from chainer import function, functions, links, optimizer
from chainer import Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import math
from chainer import initializers

import util
from util import PADDING, UNKWORD

import six


to_cpu = chainer.cuda.to_cpu


class BiLSTM(chainer.Chain):

    def __init__(self, n_vocab=None, emb_dim=100, hidden_dim=200,
                 init_emb=None, use_dropout=0.33, n_layers=1,
                 n_label=0, use_crf=True):
        # feature_dim = emb_dim + add_dim + pos_dim
        feature_dim = emb_dim
        super(BiLSTM, self).__init__(
            word_embed=L.EmbedID(n_vocab, emb_dim, ignore_label=-1),
            bi_lstm=L.NStepBiLSTM(n_layers=n_layers, in_size=feature_dim,
                                  out_size=hidden_dim, dropout=use_dropout,
                                  use_cudnn=True),
            output_layer=L.Linear(hidden_dim * 2, n_label),
        )
        # if n_pos:
        #     pos_embed = L.EmbedID(n_pos, pos_dim, ignore_label=-1)
        #     self.add_link('pos_embed', pos_embed)

        if use_crf:
            self.lossfun = L.CRF1d(n_label=n_label)

        # self.n_pos = n_pos
        self.hidden_dim = hidden_dim
        self.train = True
        self.use_dropout = use_dropout
        self.n_layers = n_layers

        # Forget gate bias => 1.0
        # MEMO: Values 1 and 5 reference the forget gate.
        for w in self.bi_lstm:
            w.b1.data[:] = 1.0
            w.b5.data[:] = 1.0

    def set_train(self, train):
        self.train = train

    def predict(self, y_list, ts):

        predict_list = []
        cnt = 0
        for n_len in n_length:
            pred = F.concat(y_list[cnt:cnt + n_len], axis=0)
            predict_list.append(pred)
            cnt += n_len

        inds_trans = [inds[i] for i in inds]
        hs = [predict_list[i] for i in inds]
        ts_original = [self.xp.array(t[i], self.xp.int32) for i in inds]

        hs = F.transpose_sequence(hs)
        ts = F.transpose_sequence(ts_original)
        loss = self.lossfun(hs, ts)
        #
        # inds = np.argsort([-len(_x) for _x in ts]).astype('i')
        # inds_trans = [inds[i] for i in inds]
        # hs_sorted = [output_list[i] for i in inds]
        # ts_sorted = [ts[i] for i in inds]
        #
        # hs_trans = F.transpose_sequence(hs_sorted)
        # ts_trans = F.transpose_sequence(ts_sorted)
        #
        # _, predicts_trans = self.lossfun.argmax(hs_trans)
        #
        # loss = self.lossfun(hs_trans, ts_trans)

        predicts = F.transpose_sequence(predicts_trans)
        gold_predict_pairs = []
        for pred, gold in zip(predicts, ts_sorted):
            pred = to_cpu(pred.data)
            gold = to_cpu(gold)
            # print pred, gold
            gold_predict_pairs.append([gold, pred])

        return gold_predict_pairs, loss

    def __call__(self, x_data, add_pos=None, add_h=None):
        hx = None
        cx = None
        n_length = [len(_x) for _x in x_data]

        xs = []
        for i, x in enumerate(x_data):
            x = Variable(x, volatile=not self.train)
            x = self.word_embed(x)
            # if self.n_pos:
            #     pos_vec = self.pos_embed(add_pos[i])
            #     x = F.concat([x, pos_vec], axis=1)
            x = F.dropout(x, ratio=self.use_dropout, train=self.train)
            xs.append(x)

        _hy_f, _cy_f, h_vecs = self.bi_lstm(hx=hx, cx=cx, xs=xs,
                                            train=self.train)

        h_vecs = F.concat(h_vecs, axis=0)
        if self.use_dropout:
            h_vecs = F.dropout(h_vecs, ratio=self.use_dropout,
                               train=self.train)

        # Label Predict
        output = self.output_layer(h_vecs)
        output_list = F.split_axis(output, output.data.shape[0], axis=0)
        # output_list = F.split_axis(output, np.cumsum(n_length[:-1]), axis=0)
        # print 'output:', output.shape
        # print [_.shape[0] for _ in output_list], sum([_.shape[0] for _ in output_list])
        return output_list
