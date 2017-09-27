
import os
os.environ["CHAINER_SEED"] = "1234"

import random
import numpy as np
random.seed(1234)
np.random.seed(1234)

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers
import chainer.functions as F

from bi_lstm import BiLSTM_CNN_CRF

import util
from util import PADDING, UNKWORD


import logging
logger = logging.getLogger(__name__)

to_cpu = chainer.cuda.to_cpu

import os.path

version = chainer.__version__


def my_cudnn(cudnn_flag):
    if version >= '2.0':
        if cudnn_flag:
            chainer.config.use_cudnn = 'always'
            # chainer.config.cudnn_deterministic = True
        else:
            chainer.config.use_cudnn = 'never'


def run(data_file, is_train=False, **args):
    is_test = not is_train
    batchsize = args['batchsize']
    model_name = args['model_name']
    optimizer_name = args['optimizer']
    save_dir = args['save_dir']
    print args
    if save_dir[-1] != '/':
        save_dir = save_dir + '/'

    # TODO: check save_dir exist
    if not os.path.isdir(save_dir):
        err_msg = 'There is no dir : {}\n'.format(save_dir)
        err_msg += '##############################\n'
        err_msg += '## Please followiing: \n'
        err_msg += '## $ mkdir {}\n'.format(save_dir)
        err_msg += '##############################\n'
        raise ValueError(err_msg)

    save_name = args['save_name']
    save_name = save_dir + save_name

    xp = cuda.cupy if args['gpu'] >= 0 else np
    efficient_gpu = False
    if args['gpu'] >= 0:
        cuda.get_device(args['gpu']).use()
        xp.random.seed(1234)
        efficient_gpu = args.get('efficient_gpu', False)

    def to_gpu(x):
        if args['gpu'] >= 0:
            return chainer.cuda.to_gpu(x)
        return x

    # load files
    dev_file = args['dev_file']
    test_file = args['test_file']
    delimiter = args['delimiter']
    input_idx = map(int, args['input_idx'].split(','))
    output_idx = map(int, args['output_idx'].split(','))
    word_input_idx = input_idx[0]  # NOTE: word_idx is first column!
    additional_input_idx = input_idx[1:]
    sentences_train = []
    if is_train:
        sentences_train = util.read_conll_file(filename=data_file,
                                               delimiter=delimiter)
        if len(sentences_train) == 0:
            s = str(len(sentences_train))
            err_msg = 'Invalid training sizes: {} sentences. '.format(s)
            raise ValueError(err_msg)
    else:
        # Predict
        if len(input_idx) == 1:
            # raw text format
            sentences_train = util.read_raw_file(filename=data_file,
                                                 delimiter=u' ')
        else:
            # conll format
            sentences_train = util.read_conll_file(filename=data_file,
                                                   delimiter=delimiter)

    # sentences_train = sentences_train[:100]

    sentences_dev = []
    sentences_test = []
    if dev_file:
        sentences_dev = util.read_conll_file(dev_file, delimiter=delimiter)
    if test_file:
        sentences_test = util.read_conll_file(test_file, delimiter=delimiter)

    # Additional setup
    vocab_adds = []
    for ad_feat_id in additional_input_idx:
        sentences_additional_train = [[feat_obj[ad_feat_id] for feat_obj in sentence]
                                      for sentence in sentences_train]
        vocab_add = util.build_vocab(sentences_additional_train)
        vocab_adds.append(vocab_add)

    save_vocab = save_name + '.vocab'
    save_vocab_char = save_name + '.vocab_char'
    save_tags_vocab = save_name + '.vocab_tag'
    save_train_config = save_name + '.train_config'

    # TODO: check unkown pos tags
    # TODO: compute unk words

    if is_train:
        sentences_words_train = [[w_obj[word_input_idx] for w_obj in sentence]
                                 for sentence in sentences_train]
        vocab = util.build_vocab(sentences_words_train)
        vocab_char = util.build_vocab(util.flatten(sentences_words_train))
        vocab_tags = util.build_tag_vocab(sentences_train)

    elif is_test:
        vocab = util.load_vocab(save_vocab)
        vocab_char = util.load_vocab(save_vocab_char)
        vocab_tags = util.load_vocab(save_tags_vocab)
        vocab_adds = []
        for i, idx in enumerate(additional_input_idx):
            save_additional_vocab = save_name + '.vocab_additional_' + str(i)
            vocab_add = util.load_vocab(save_additional_vocab)
            vocab_adds.append(vocab_add)

    if args.get('word_emb_file', False):
        # set Pre-trained embeddings
        # emb_file = './emb/glove.6B.100d.txt'
        emb_file = args['word_emb_file']
        word_emb_vocab_type = args.get('word_emb_vocab_type')

        def assert_word_emb_shape(shape1, shape2):
            err_msg = '''Pre-trained embedding size is not equal to `--n_word_emb` ({} != {})'''
            if shape1 != shape2:
                err_msg = err_msg.format(str(shape1), str(shape2))
                raise ValueError(err_msg)

        def assert_no_emb(word_vecs):
            err_msg = '''There is no-embeddings! Please check your file `--word_emb_file`'''
            if word_vecs.shape[0] == 0:
                raise ValueError(err_msg)

        if word_emb_vocab_type == 'replace_all':
            # replace all vocab by Pre-trained embeddings
            word_vecs, vocab_glove = util.load_glove_embedding_include_vocab(emb_file)
            vocab = vocab_glove
        elif word_emb_vocab_type == 'replace_only':
            word_ids, word_vecs = util.load_glove_embedding(emb_file, vocab)
            assert_no_emb(word_vecs)

        elif word_emb_vocab_type == 'additional':
            word_vecs, vocab_glove = util.load_glove_embedding_include_vocab(emb_file)
            additional_vecs = []
            for word, word_idx in sorted(vocab_glove.items(), key=lambda x: x[1]):
                if word not in vocab:
                    vocab[word] = len(vocab)
                    additional_vecs.append(word_vecs[word_idx])
            additional_vecs = np.array(additional_vecs, dtype=np.float32)

    if args.get('vocab_file', False):
        vocab_file = args['vocab_file']
        vocab = util.load_vocab(vocab_file)

    if args.get('vocab_char_file', False):
        vocab_char_file = args['vocab_char_file']
        vocab_char = util.load_vocab(vocab_char_file)

    vocab_tags_inv = dict((v, k) for k, v in vocab_tags.items())
    PAD_IDX = vocab[PADDING]
    UNK_IDX = vocab[UNKWORD]

    CHAR_PAD_IDX = vocab_char[PADDING]
    CHAR_UNK_IDX = vocab_char[UNKWORD]

    tmp_xp = xp
    if efficient_gpu:
        tmp_xp = np  # use CPU (numpy)

    def parse_to_word_ids(sentences, word_input_idx, vocab):
        return util.parse_to_word_ids(sentences, xp=tmp_xp, vocab=vocab,
                                      UNK_IDX=UNK_IDX, idx=word_input_idx)

    def parse_to_char_ids(sentences):
        return util.parse_to_char_ids(sentences, xp=tmp_xp, vocab_char=vocab_char,
                                      UNK_IDX=CHAR_UNK_IDX, idx=word_input_idx)

    def parse_to_tag_ids(sentences):
        return util.parse_to_tag_ids(sentences, xp=tmp_xp, vocab=vocab_tags,
                                     UNK_IDX=-1, idx=-1)

    x_train = parse_to_word_ids(sentences_train, word_input_idx, vocab)
    x_char_train = parse_to_char_ids(sentences_train)
    y_train = parse_to_tag_ids(sentences_train)
    x_train_additionals = [parse_to_word_ids(sentences_train, ad_feat_id, vocab_adds[i])
                           for i, ad_feat_id in enumerate(additional_input_idx)]

    x_dev = parse_to_word_ids(sentences_dev, word_input_idx, vocab)
    x_char_dev = parse_to_char_ids(sentences_dev)
    y_dev = parse_to_tag_ids(sentences_dev)
    x_dev_additionals = [parse_to_word_ids(sentences_dev, ad_feat_id, vocab_adds[i])
                         for i, ad_feat_id in enumerate(additional_input_idx)]

    y_dev_cpu = [[w[-1] for w in sentence]
                 for sentence in sentences_dev]
    # tag_names = []
    tag_names = list(set([tag[2:] if len(tag) >= 2 else tag[0] for tag in vocab_tags.keys()]))

    x_test = parse_to_word_ids(sentences_test, word_input_idx, vocab)
    x_char_test = parse_to_char_ids(sentences_test)
    y_test = parse_to_tag_ids(sentences_test)
    x_test_additionals = [parse_to_word_ids(sentences_test, ad_feat_id, vocab_adds[i])
                          for i, ad_feat_id in enumerate(additional_input_idx)]

    cnt_train_unk = sum([tmp_xp.sum(d == UNK_IDX) for d in x_train])
    cnt_train_word = sum([d.size for d in x_train])
    unk_train_unk_rate = float(cnt_train_unk) / cnt_train_word

    cnt_dev_unk = sum([tmp_xp.sum(d == UNK_IDX) for d in x_dev])
    cnt_dev_word = sum([d.size for d in x_dev])
    unk_dev_unk_rate = float(cnt_dev_unk) / max(cnt_dev_word, 1)

    logging.info('train:' + str(len(x_train)))
    logging.info('dev  :' + str(len(x_dev)))
    logging.info('test :' + str(len(x_test)))
    logging.info('vocab     :' + str(len(vocab)))
    logging.info('vocab_tags:' + str(len(vocab_tags)))
    logging.info('unk count (train):' + str(cnt_train_unk))
    logging.info('unk rate  (train):' + str(unk_train_unk_rate))
    logging.info('cnt all words (train):' + str(cnt_train_word))
    logging.info('unk count (dev):' + str(cnt_dev_unk))
    logging.info('unk rate  (dev):' + str(unk_dev_unk_rate))
    logging.info('cnt all words (dev):' + str(cnt_dev_word))
    # show model config
    logging.info('######################')
    logging.info('## Model Config')
    logging.info('model_name:' + str(model_name))
    logging.info('batchsize:' + str(batchsize))
    logging.info('optimizer:' + str(optimizer_name))
    # Save model config
    logging.info('######################')
    logging.info('## Model Save Config')
    logging.info('save_dir :' + str(save_dir))

    # save vocab
    logging.info('save_vocab        :' + save_vocab)
    logging.info('save_vocab_char   :' + save_vocab_char)
    logging.info('save_tags_vocab   :' + save_tags_vocab)
    logging.info('save_train_config :' + save_train_config)

    init_emb = None

    if is_train:
        util.write_vocab(save_vocab, vocab)
        util.write_vocab(save_vocab_char, vocab_char)
        util.write_vocab(save_tags_vocab, vocab_tags)
        util.write_vocab(save_train_config, args)

        for i, vocab_add in enumerate(vocab_adds):
            save_additional_vocab = save_name + '.vocab_additional_' + str(i)
            util.write_vocab(save_additional_vocab, vocab_add)

    n_vocab_add = [len(_vadd) for _vadd in vocab_adds]

    net = BiLSTM_CNN_CRF(n_vocab=len(vocab), n_char_vocab=len(vocab_char),
                         emb_dim=args['n_word_emb'],
                         hidden_dim=args['n_hidden'],
                         n_layers=args['n_layer'], init_emb=init_emb,
                         char_input_dim=args['n_char_emb'],
                         char_hidden_dim=args['n_char_hidden'],
                         n_label=len(vocab_tags),
                         n_add_feature_dim=args['n_add_feature_emb'],
                         n_add_feature=len(n_vocab_add),
                         n_vocab_add=n_vocab_add,
                         use_cudnn=args['use_cudnn'])
    my_cudnn(args['use_cudnn'])

    if args.get('word_emb_file', False):

        if word_emb_vocab_type == 'replace_all':
            # replace all vocab by Pre-trained embeddings
            assert_word_emb_shape(word_vecs.shape[1], net.word_embed.W.shape[1])
            net.word_embed.W.data = word_vecs[:]
        elif word_emb_vocab_type == 'replace_only':
            assert_no_emb(word_vecs)
            assert_word_emb_shape(word_vecs.shape[1], net.word_embed.W.shape[1])
            net.word_embed.W.data[word_ids] = word_vecs[:]

        elif word_emb_vocab_type == 'additional':
            assert_word_emb_shape(word_vecs.shape[1], net.word_embed.W.shape[1])
            v_size = additional_vecs.shape[0]
            net.word_embed.W.data[-v_size:] = additional_vecs[:]

    if args.get('return_model', False):
        return net

    if args['gpu'] >= 0:
        net.to_gpu()

    init_alpha = args['init_lr']
    if optimizer_name == 'adam':
        opt = optimizers.Adam(alpha=init_alpha, beta1=0.9, beta2=0.9)
    elif optimizer_name == 'adadelta':
        opt = optimizers.AdaDelta()
    if optimizer_name == 'sgd_mom':
        opt = optimizers.MomentumSGD(lr=init_alpha, momentum=0.9)
    if optimizer_name == 'sgd':
        opt = optimizers.SGD(lr=init_alpha)

    opt.setup(net)
    opt.add_hook(chainer.optimizer.GradientClipping(5.0))

    def eval_loop(x_data, x_char_data, y_data, x_train_additionals=[]):
        # dev or test
        net.set_train(train=False)
        iteration_list = range(0, len(x_data), batchsize)
        # perm = np.random.permutation(len(x_data))
        sum_loss = 0.0
        predict_lists = []
        for i_index, index in enumerate(iteration_list):
            x = x_data[index:index + batchsize]
            x_char = x_char_data[index:index + batchsize]
            target_y = y_data[index:index + batchsize]

            if efficient_gpu:
                x = [to_gpu(_) for _ in x]
                x_char = [[to_gpu(_) for _ in words] for words in x_char]
                target_y = [to_gpu(_) for _ in target_y]

            x_additional = []
            if len(x_train_additionals):
                x_additional = [[to_gpu(_) for _ in x_ad[index:index + batchsize]]
                                for x_ad in x_train_additionals]

            output = net(x_data=x, x_char_data=x_char, x_additional=x_additional)
            predict, loss = net.predict(output, target_y)

            sum_loss += loss.data
            predict_lists.extend(predict)

        _, predict_tags = zip(*predict_lists)
        predicted_results = []
        for predict in predict_tags:
            predicted = [vocab_tags_inv[tag_idx] for tag_idx in to_cpu(predict)]
            predicted_results.append(predicted)

        return predict_lists, sum_loss, predicted_results

    if args['model_filename']:
        model_filename = args['model_filename']
        serializers.load_hdf5(model_filename, net)

    if is_test:
        # predict
        # model_filename = args['model_filename']
        # model_filename = save_dir + model_filename
        # serializers.load_hdf5(model_filename, net)
        vocab_tags_inv = dict([(v, k) for k, v in vocab_tags.items()])
        x_predict = x_train
        x_char_predict = x_char_train
        x_additionals = x_train_additionals
        y_predict = y_train

        if dev_file:
            predict_dev, loss_dev, predict_dev_tags = eval_loop(
                x_dev, x_char_dev, y_dev, x_dev_additionals)
            gold_predict_pairs = [y_dev_cpu, predict_dev_tags]
            result, phrase_info = util.conll_eval(
                gold_predict_pairs, flag=False, tag_class=tag_names)
            all_result = result['All_Result']
            print 'all_result:', all_result

        predict_pairs, _, _tmp = eval_loop(x_predict, x_char_predict, y_predict, x_additionals)
        _, predict_tags = zip(*predict_pairs)
        predicted_output = args['predicted_output']
        predicted_results = []
        for predict in predict_tags:
            predicted = [vocab_tags_inv[tag_idx] for tag_idx in to_cpu(predict)]
            predicted_results.append(predicted)

        f = open(predicted_output, 'w')
        for predicted in predicted_results:
            for tag in predicted:
                f.write(tag + '\n')
            f.write('\n')
        f.close()

        return False

    tmax = args['max_iter']
    t = 0.0
    prev_dev_accuracy = 0.0
    prev_dev_f = 0.0
    for epoch in xrange(args['max_iter']):

        # train
        net.set_train(train=True)
        iteration_list = range(0, len(x_train), batchsize)
        perm = np.random.permutation(len(x_train))
        sum_loss = 0.0
        predict_train = []
        for i_index, index in enumerate(iteration_list):
            data = [(x_train[i], x_char_train[i], y_train[i])
                    for i in perm[index:index + batchsize]]
            x, x_char, target_y = zip(*data)

            x_additional = []
            if len(x_train_additionals):
                x_additional = [[to_gpu(x_ad[add_i]) for add_i in perm[index:index + batchsize]]
                                for x_ad in x_train_additionals]

            if efficient_gpu:
                x = [to_gpu(_) for _ in x]
                x_char = [[to_gpu(_) for _ in words] for words in x_char]
                target_y = [to_gpu(_) for _ in target_y]

            output = net(x_data=x, x_char_data=x_char, x_additional=x_additional)
            predict, loss = net.predict(output, target_y)

            # loss
            sum_loss += loss.data

            # update
            net.zerograds()
            loss.backward()
            opt.update()

            predict_train.extend(predict)

        # Evaluation
        train_accuracy = util.eval_accuracy(predict_train)

        logging.info('epoch:' + str(epoch))
        logging.info(' [train]')
        logging.info('  loss     :' + str(sum_loss))
        logging.info('  accuracy :' + str(train_accuracy))

        # Dev
        predict_dev, loss_dev, predict_dev_tags = eval_loop(
            x_dev, x_char_dev, y_dev, x_dev_additionals)

        gold_predict_pairs = [y_dev_cpu, predict_dev_tags]
        result, phrase_info = util.conll_eval(gold_predict_pairs, flag=False, tag_class=tag_names)
        all_result = result['All_Result']

        # Evaluation
        dev_accuracy = util.eval_accuracy(predict_dev)
        logging.info(' [dev]')
        logging.info('  loss     :' + str(loss_dev))
        logging.info('  accuracy :' + str(dev_accuracy))
        logging.info('  f_measure :' + str(all_result[-1]))

        dev_f = all_result[-1]

        if prev_dev_f <= dev_f:
            logging.info(' [update best model on dev set!]')
            dev_list = [prev_dev_f, dev_f]
            dev_str = '       ' + ' => '.join(map(str, dev_list))
            logging.info(dev_str)
            prev_dev_f = dev_f

            # Save model
            model_filename = save_name + '_epoch' + str(epoch)
            serializers.save_hdf5(model_filename + '.model', net)
            serializers.save_hdf5(model_filename + '.state', opt)
