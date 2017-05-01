import os
import sys
import re
import numpy as np

from itertools import chain

pattern_num = re.compile(r'[0-9]')
CHAR_PADDING = u"_"
UNKWORD = u"UNKNOWN"
PADDING = u"PADDING"
BOS = u"<BOS>"


def flatten(l):
    return list(chain.from_iterable(l))


def replace_num(text):
    return pattern_num.sub(u'0', text)


def build_vocab(dataset, min_count=0):
    vocab = {}
    vocab[PADDING] = len(vocab)
    vocab[UNKWORD] = len(vocab)
    vocab_cnt = {}
    for d in dataset:
        for w in d:
            vocab_cnt[w] = vocab_cnt.get(w, 0) + 1

    for w, cnt in sorted(vocab_cnt.items(), key=lambda x: x[1], reverse=True):
        if cnt >= min_count:
            vocab[w] = len(vocab)

    return vocab


def build_tag_vocab(dataset, tag_idx=-1):
    pos_tags = list(set(flatten([[w[tag_idx] for w in word_objs]
                                 for word_objs in dataset])))
    pos_tags = sorted(pos_tags)
    vocab = {}
    for pos in pos_tags:
        vocab[pos] = len(vocab)
    return vocab


def read_conll_file(filename, delimiter=u'\t'):
    sentence = []
    sentences = []
    n_features = -1
    for line_idx, l in enumerate(open(filename, 'r')):
        l_split = l.strip().decode('utf-8').split(delimiter)
        if len(l_split) <= 1:
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        else:
            if n_features == -1:
                n_features = len(l_split)

            if n_features != len(l_split):
                raise ValueError('Invalid input feature sizes: "%s". Please check at line [%s]'
                                 % (str(len(l_split)), str(len(line_idx))))
            sentence.append(l_split)
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences


def load_glove_embedding(filename, vocab):
    word_ids = []
    word_vecs = []
    for i, l in enumerate(open(filename)):
        l = l.decode('utf-8').split(u' ')
        word = l[0].lower()

        if word in vocab:
            word_ids.append(vocab.get(word))
            vec = l[1:]
            vec = map(float, vec)
            word_vecs.append(vec)
    word_ids = np.array(word_ids, dtype=np.int32)
    word_vecs = np.array(word_vecs, dtype=np.float32)
    return word_ids, word_vecs
