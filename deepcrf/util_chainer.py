#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import Variable
import chainer.functions as F

version = chainer.__version__


def my_variable(x, volatile):
    if version < '2.0':
        # v1.24.0
        return Variable(x, volatile=volatile)
    else:
        # v2.0
        return Variable(x)


def my_dropout(x, ratio, train):
    if version < '2.0':
        return F.dropout(x, ratio=ratio, train=train)
    else:
        # v2.0
        return F.dropout(x, ratio=ratio)


def my_set_train(train):
    if version >= '2.0':
        chainer.config.train = train


def my_rnn_link(rnn_link, n_layers, feature_dim, hidden_dim, use_dropout, use_cudnn):
    if version < '2.0':
        return rnn_link(n_layers=n_layers, in_size=feature_dim,
                        out_size=hidden_dim, dropout=use_dropout,
                        use_cudnn=use_cudnn)
    else:
        # v2.0
        return rnn_link(n_layers=n_layers, in_size=feature_dim,
                        out_size=hidden_dim, dropout=use_dropout)
