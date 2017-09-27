import click
import logging

import main
import evaluate


@click.group()
def cli():
    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


@cli.command()
@click.argument('train_file', type=click.Path(exists=True))
@click.option('--save_dir', type=str, default='save_model_dir',
              help='save model  dir')
@click.option('--model_name', type=str, default='bilstm-cnn-crf',
              help="select from [bilstm-cnn-crf, bilstm-cnn]")
@click.option('--batchsize', type=int, default=32, help='batch size')
@click.option('--max_iter', type=int, default=50, help='max iterations (default: 50)')
@click.option('--optimizer', type=str, default='adam',
              help="select from [adam, adadelta, sgd, sgd_mom]")
@click.option('--init_lr', type=float, default=0.001, help='Initial Learning rate (default: 0.001)')
@click.option('--weight_decay', type=float, default=0.0)
@click.option('--use_lr_decay', type=int, default=0)
@click.option('--use_crf', type=int, default=1, help='use CRF flag.')
@click.option('--n_layer', type=int, default=1)
@click.option('--n_hidden', type=int, default=200)
@click.option('--n_vocab_min_cnt', type=int, default=0,
              help='min count of vocab.')
@click.option('--n_word_emb', type=int, default=100,
              help='word embedding size.')
@click.option('--n_add_feature_emb', type=int, default=100,
              help='additional feature embedding size.')
@click.option('--n_char_emb', type=int, default=30,
              help='character embedding size.')
@click.option('--n_char_hidden', type=int, default=30,
              help='character hidden vector size.')
@click.option('--dropout_rate', type=float, default=0.33)
@click.option('--gpu', type=int, default=-1,
              help='gpu ID. when gpu=-1 use CPU mode.')
@click.option('--word_emb_file', type=click.Path())
@click.option('--word_emb_vocab_type', type=str, default='replace_all',
              help="select from [replace_all, replace_only, additional]")
@click.option('--vocab_file', type=click.Path())
@click.option('--vocab_char_file', type=click.Path())
@click.option('--dev_file', type=click.Path())
@click.option('--test_file', type=click.Path())
@click.option('--model_filename', type=click.Path())
@click.option('--input_idx', type=str, default='0', help='input_idx for features.')
@click.option('--output_idx', type=str, default='-1', help='output_idx for predicting.')
@click.option('--delimiter', type=str, default='\t',
              help='delimiter string')
@click.option('--save_name', type=str, default='bilstm-cnn-crf_adam', help='save_name')
@click.option('--use_cudnn', type=int, default=1, help='use_cudnn = 0 or 1')
@click.option('--efficient_gpu', type=int, default=1, help='efficient_gpu (if efficient_gpu == 1, it needs small GPU memory)')
def train(train_file, **args):
    # load input_file
    main.run(train_file, is_train=True, **args)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--save_dir', type=str, default='save_model_dir',
              help='save model  dir')
@click.option('--model_name', type=str, default='bilstm-cnn-crf',
              help="select from [bilstm-cnn-crf, bilstm-cnn]")
@click.option('--batchsize', type=int, default=32, help='batch size')
@click.option('--max_iter', type=int, default=50, help='max iterations (default: 50)')
@click.option('--optimizer', type=str, default='adam',
              help="select from [adam, adadelta, sgd, sgd_mom]")
@click.option('--init_lr', type=float, default=0.001, help='Initial Learning rate (default: 0.001)')
@click.option('--weight_decay', type=float, default=0.0)
@click.option('--use_lr_decay', type=int, default=0)
@click.option('--use_crf', type=int, default=1, help='use CRF flag.')
@click.option('--n_layer', type=int, default=1)
@click.option('--n_hidden', type=int, default=200)
@click.option('--n_vocab_min_cnt', type=int, default=0,
              help='min count of vocab.')
@click.option('--n_word_emb', type=int, default=100,
              help='word embedding size.')
@click.option('--n_add_feature_emb', type=int, default=100,
              help='additional feature embedding size.')
@click.option('--n_char_emb', type=int, default=30,
              help='character embedding size.')
@click.option('--n_char_hidden', type=int, default=30,
              help='character hidden vector size.')
@click.option('--dropout_rate', type=float, default=0.33)
@click.option('--gpu', type=int, default=-1,
              help='gpu ID. when gpu=-1 use CPU mode.')
@click.option('--word_emb_file', type=click.Path())
@click.option('--word_emb_vocab_type', type=str, default='replace_all',
              help="select from [replace_all, replace_only, additional]")
@click.option('--dev_file', type=click.Path())
@click.option('--test_file', type=click.Path())
@click.option('--vocab_file', type=click.Path())
@click.option('--vocab_char_file', type=click.Path())
@click.option('--delimiter', type=str, default='\t',
              help='delimiter string')
@click.option('--save_name', type=str, default='bilstm-cnn-crf_adam', help='save_name')
@click.option('--predicted_output', type=str, default='',
              help='predicted_output')
@click.option('--model_filename', type=click.Path())
@click.option('--input_idx', type=str, default='0', help='input_idx for features.')
@click.option('--output_idx', type=str, default='-1', help='output_idx for predicting.')
@click.option('--use_cudnn', type=int, default=1, help='use_cudnn = 0 or 1')
@click.option('--efficient_gpu', type=int, default=1, help='efficient_gpu (if efficient_gpu == 1, it needs small GPU memory)')
def predict(input_file, **args):
    main.run(input_file, is_train=False, **args)


@cli.command()
@click.argument('gold_file', type=click.Path(exists=True))
@click.argument('predicted_file', type=click.Path(exists=True))
@click.option('--tag_type', type=str, default='BIOES', help='select from [BIO, BIOES]')
def eval(gold_file, predicted_file, **args):
    evaluate.run(gold_file, predicted_file, **args)
