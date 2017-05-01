
import os
os.environ["CHAINER_SEED"] = "1234"

import random
import numpy as np
random.seed(1234)
np.random.seed(1234)

import chainer
from chainer import cuda
from chainer import optimizers
import chainer.functions as F

from bi_lstm import BiLSTM

import util
from util import PADDING, UNKWORD


import logging
logger = logging.getLogger(__name__)

to_cpu = chainer.cuda.to_cpu


def train(train_file, **args):
    batchsize = args['batchsize']
    print args

    xp = cuda.cupy if args['gpu'] >= 0 else np
    if args['gpu'] >= 0:
        cuda.get_device(args['gpu']).use()
        xp.random.seed(1234)

    # load files
    dev_file = args['dev_file']
    test_file = args['test_file']
    sentences_train = util.read_conll_file(train_file, delimiter=u' ')
    sentences_dev = []
    sentences_test = []
    if dev_file:
        sentences_dev = util.read_conll_file(dev_file)
    if test_file:
        sentences_test = util.read_conll_file(test_file)

    # TODO: check unkown pos tags
    # TODO: compute unk words
    sentences_words_train = [w_obj[0] for w_obj in sentences_train]
    vocab = util.build_vocab(sentences_words_train)
    vocab_tags = util.build_tag_vocab(sentences_train)

    PAD_IDX = vocab[PADDING]
    UNK_IDX = vocab[UNKWORD]

    def parse_to_word_ids(sentences):
        x_data = [xp.array([vocab.get(w[0].lower(), UNK_IDX)
                            for w in sentence], dtype=xp.int32)
                  for sentence in sentences]
        return x_data

    def parse_to_tag_ids(sentences):
        x_data = [xp.array([vocab_tags.get(w[-1], -1)
                            for w in sentence], dtype=xp.int32)
                  for sentence in sentences]
        return x_data

    x_train = parse_to_word_ids(sentences_train)
    y_train = parse_to_tag_ids(sentences_train)

    x_dev = parse_to_word_ids(sentences_dev)
    y_dev = parse_to_tag_ids(sentences_dev)

    x_test = parse_to_word_ids(sentences_test)
    y_test = parse_to_tag_ids(sentences_test)

    cnt_train_unk = sum([xp.sum(d == UNK_IDX) for d in x_train])
    cnt_train_word = sum([d.size for d in x_train])
    unk_train_unk_rate = float(cnt_train_unk) / cnt_train_word

    cnt_dev_unk = sum([xp.sum(d == UNK_IDX) for d in x_dev])
    cnt_dev_word = sum([d.size for d in x_dev])
    unk_dev_unk_rate = float(cnt_dev_unk) / max(cnt_dev_word, 1)

    logging.info('vocab=' + str(len(vocab)))
    logging.info('train=' + str(len(x_train)))
    logging.info('dev=' + str(len(x_dev)))
    logging.info('test=' + str(len(x_test)))
    logging.info('vocab_tags=' + str(len(vocab_tags)))
    logging.info('unk count [train]=' + str(cnt_train_unk))
    logging.info('unk rate [train]=' + str(unk_train_unk_rate))
    logging.info('cnt all words [train]=' + str(cnt_train_word))
    logging.info('unk count [dev]=' + str(cnt_dev_unk))
    logging.info('unk rate [dev]=' + str(unk_dev_unk_rate))
    logging.info('cnt all words [dev]=' + str(cnt_dev_word))

    net = BiLSTM(n_vocab=len(vocab), emb_dim=args['n_word_emb'],
                 hidden_dim=args['n_hidden'],
                 n_layers=args['n_layer'], init_emb=None,
                 n_label=len(vocab_tags))

    if args['word_emb_file']:
        # glove
        emb_file = './emb/glove.6B.100d.txt'
        word_ids, word_vecs = util.load_glove_embedding(emb_file, vocab)
        net.word_embed.W.data[word_ids] = word_vecs

    if args['gpu'] >= 0:
        net.to_gpu()

    init_alpha = args['init_lr']
    opt = optimizers.Adam(alpha=init_alpha, beta1=0.9, beta2=0.9, eps=1e-12)
    opt.setup(net)
    opt.add_hook(chainer.optimizer.GradientClipping(5.0))
    tmax = args['max_iter']
    t = 0.0
    for epoch in xrange(args['max_iter']):

        # train
        net.set_train(train=True)
        iteration_list = range(0, len(x_train), batchsize)
        perm = np.random.permutation(len(x_train))
        sum_loss = 0.0
        predict_lists = []
        for i_index, index in enumerate(iteration_list):
            data = [(x_train[i], y_train[i])
                    for i in perm[index:index + batchsize]]
            x, target_y = zip(*data)

            output = net(x_data=x)
            predict, loss = net.predict(output, target_y)

            # loss
            sum_loss += loss.data

            # update
            net.zerograds()
            loss.backward()
            opt.update()

            # predict
            predict_lists.append(predict)

            # annealing
            t += 1
            alpha_update = min(init_alpha * (0.75 ** (t / max(t, tmax))), init_alpha)
            opt.alpha = alpha_update

        logging.info('epoch:' + str(epoch))
        logging.info(' loss:' + str(sum_loss))
        logging.info(' alpha:' + str(opt.alpha))
        # logging.info(' t:' + str(t))

        # output file
        # predict_lists = sorted(predict_lists, key=lambda x: x[0])
        # predict_lists = [(_[1], _[2]) for _ in predict_lists]
        # output_file = './predict/predict_train_' + args.saveopt + '.txt'
        # UAS_f1, LAS_f1 = util.output_predict_file(output_file, train_file,
        #                                           predict_lists, vocab_dep_inv)

        # # Dev
        # biaffine.set_train(train=False)
        # iteration_list = range(0, len(x_dev), batchsize)
        # predict_lists = []
        # for i_index, index in enumerate(iteration_list):
        #     x = x_dev[index:index + batchsize]
        #     x_pos = x_pos_dev[index:index + batchsize]
        #     target_y = y_dev[index:index + batchsize]
        #     target_y_label = y_label_dev[index:index + batchsize]
        #
        #     target_y, target_y_offset, target_y_label = biaffine.pre_padding_y(
        #         target_y, target_y_label)
        #
        #     attention_batch = biaffine(x_data=x, x_pos_data=x_pos)
        #
        #     # predict
        #     predict_result = biaffine.predict(attention_batch)
        #     _, predict_y_offset, _ = biaffine.pre_padding_y(predict_result, None)
        #     h_labels_output = biaffine.output_labels(predict_y_offset)
        #     predict_labels = biaffine.predict_labels(h_labels_output)
        #
        #     # save predicted result
        #     for predict, predict_label in zip(predict_result, predict_labels):
        #         item = [to_cpu(predict), to_cpu(predict_label)]
        #         predict_lists.append(item)
        #
        # # output file
        # output_file = './predict/predict_dev_' + args.saveopt + '.txt'
        # UAS_f1, LAS_f1 = util.output_predict_file(output_file, dev_file,
        #                                           predict_lists, vocab_dep_inv)
        #
        # logging.info(' [dev]:')
        # logging.info(' UAS_f1:' + str(UAS_f1))
        # logging.info(' LAS_f1:' + str(LAS_f1))
