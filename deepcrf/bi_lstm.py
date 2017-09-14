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

from cnn import CharCNNEncoder
import util
from util import PADDING, UNKWORD

import six

to_cpu = chainer.cuda.to_cpu


class BiLSTM_CNN_CRF(chainer.Chain):

    def __init__(self, n_vocab=None, n_char_vocab=None, emb_dim=100,
                 hidden_dim=200, init_emb=None, use_dropout=0.33, n_layers=1,
                 n_label=0, use_crf=True, use_bi=True, char_input_dim=100,
                 char_hidden_dim=100, rnn_name='bilstm', demo=False,
                 n_add_feature_dim=0, n_add_feature=0, n_vocab_add=[]):
        # feature_dim = emb_dim + add_dim + pos_dim
        n_dir = 2 if use_bi else 1
        feature_dim = emb_dim + n_add_feature_dim * n_add_feature
        self.n_add_feature_dim = n_add_feature_dim
        self.n_add_feature = n_add_feature
        use_char = False
        if n_char_vocab is not None:
            use_char = True
            feature_dim += char_hidden_dim

        rnn_names = ['bilstm', 'lstm', 'bigru', 'gru', 'birnn', 'rnn']
        rnn_links = [L.NStepBiLSTM, L.NStepLSTM, L.NStepBiGRU, L.NStepGRU,
                     L.NStepBiRNNTanh, L.NStepRNNTanh]
        if rnn_name not in rnn_names:
            candidate = ','.join(rnn_list)
            raise ValueError('Invalid RNN name: "%s". Please select from [%s]'
                             % (rnn_name, candidate))

        rnn_link = rnn_links[rnn_names.index(rnn_name)]

        super(BiLSTM_CNN_CRF, self).__init__(
            word_embed=L.EmbedID(n_vocab, emb_dim, ignore_label=-1),
            rnn=rnn_link(n_layers=n_layers, in_size=feature_dim,
                         out_size=hidden_dim, dropout=use_dropout),
            output_layer=L.Linear(hidden_dim * n_dir, n_label),
        )
        if init_emb is not None:
            self.word_embed.W.data[:] = init_emb[:]

        if use_char:

            char_cnn = CharCNNEncoder(emb_dim=char_input_dim, window_size=3,
                                      hidden_dim=char_hidden_dim,
                                      vocab_size=n_char_vocab, init_emb=None,
                                      PAD_IDX=0)
            self.add_link('char_cnn', char_cnn)

        if self.n_add_feature:
            for i in six.moves.range(self.n_add_feature):
                n_add_vocab = n_vocab_add[i]
                add_embed = L.EmbedID(n_add_vocab, n_add_feature_dim, ignore_label=-1)
                self.add_link('add_embed_' + str(i), add_embed)

        # if n_pos:
        #     pos_embed = L.EmbedID(n_pos, pos_dim, ignore_label=-1)
        #     self.add_link('pos_embed', pos_embed)

        if use_crf:
            if demo:
                import my_crf
                self.add_link('lossfun', my_crf.CRF1d(n_label=n_label))
            else:
                self.add_link('lossfun', L.CRF1d(n_label=n_label))

        # self.n_pos = n_pos
        self.hidden_dim = hidden_dim
        self.train = True
        self.use_dropout = use_dropout
        self.n_layers = n_layers
        self.use_char = use_char

        # Forget gate bias => 1.0
        # MEMO: Values 1 and 5 reference the forget gate.
        for w in self.rnn:
            w.b1.data[:] = 1.0
            w.b5.data[:] = 1.0

    def get_layer(self, name):
        return self.__getitem__(name)

    def set_train(self, train):
        self.train = train

        if self.use_char:
            self.char_cnn.set_train(train)

    def predict(self, y_list, t, compute_loss=True):

        predict_list = []
        cnt = 0
        for n_len in self.n_length:
            pred = F.concat(y_list[cnt:cnt + n_len], axis=0)
            predict_list.append(pred)
            cnt += n_len

        inds = self.inds
        # inds_trans = [inds[i] for i in inds]
        inds_rev = sorted([(i, ind) for i, ind in enumerate(inds)], key=lambda x: x[1])

        hs = [predict_list[i] for i in inds]
        ts_original = None
        if compute_loss:
            ts_original = [self.xp.array(t[i], self.xp.int32) for i in inds]

        hs = F.transpose_sequence(hs)

        loss = None
        if compute_loss and ts_original is not None:
            # loss
            ts = F.transpose_sequence(ts_original)
            loss = self.lossfun(hs, ts)

        # predict
        score, predicts_trans = self.lossfun.argmax(hs)
        predicts = F.transpose_sequence(predicts_trans)
        gold_predict_pairs = []
        if compute_loss:
            for pred, gold in zip(predicts, ts_original):
                pred = to_cpu(pred.data)
                gold = to_cpu(gold)
                gold_predict_pairs.append([gold, pred])
        else:
            for pred in predicts:
                pred = to_cpu(pred.data)
                gold_predict_pairs.append([pred])

        gold_predict_pairs = [gold_predict_pairs[e_i] for e_i, _ in inds_rev]
        self.y = gold_predict_pairs

        return gold_predict_pairs, loss

    def __call__(self, x_data, x_char_data=None, x_additional=None):
        hx = None
        cx = None
        self.n_length = [len(_x) for _x in x_data]
        self.inds = np.argsort([-len(_x) for _x in x_data]).astype('i')

        if self.use_char:
            # CharCNN
            x_char_data_flat = []
            for _ in x_char_data:
                x_char_data_flat.extend(_)
            char_vecs = self.char_cnn(x_char_data_flat)
            char_index = self.char_cnn.char_index(self.n_length)

        xs = []
        for i, x in enumerate(x_data):
            x = Variable(x)
            x = self.word_embed(x)

            if self.use_char:
                x_char = F.embed_id(char_index[i], char_vecs, ignore_label=-1)
                x = F.concat([x, x_char], axis=1)

            if x_additional:
                for add_i in six.moves.range(self.n_add_feature):
                    x_add = x_additional[add_i][i]
                    x_add = Variable(x_add)
                    add_emb_layer = self.get_layer('add_embed_' + str(add_i))
                    x_add = add_emb_layer(x_add)
                    x = F.concat([x, x_add], axis=1)

            x = F.dropout(x, ratio=self.use_dropout)
            xs.append(x)

        _hy_f, _cy_f, h_vecs = self.rnn(hx=hx, cx=cx, xs=xs,)

        h_vecs = F.concat(h_vecs, axis=0)
        if self.use_dropout:
            h_vecs = F.dropout(h_vecs, ratio=self.use_dropout)

        # Label Predict
        output = self.output_layer(h_vecs)
        output_list = F.split_axis(output, output.data.shape[0], axis=0)

        return output_list
