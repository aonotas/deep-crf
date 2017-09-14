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
              help='trained model save dir')
@click.option('--model_name', type=str, default='bilstm-cnn-crf',
              help="select from [bilstm-cnn-crf, bilstm-cnn]")
@click.option('--batchsize', type=int, default=32)
@click.option('--max_iter', type=int, default=100)
@click.option('--optimizer', type=str, default='adam',
              help="select from [adam, adadelta, sgd, sgd_mom]")
@click.option('--init_lr', type=float, default=0.01)
@click.option('--weight_decay', type=float, default=0.0)
@click.option('--use_lr_decay', type=int, default=0)
@click.option('--use_crf', type=int, default=1, help='use CRF flag.')
@click.option('--n_layer', type=int, default=1)
@click.option('--n_hidden', type=int, default=200)
@click.option('--n_vocab_min_cnt', type=int, default=0,
              help='min count of vocab.')
@click.option('--n_word_emb', type=int, default=100,
              help='word embedding size.')
@click.option('--n_char_emb', type=int, default=30,
              help='character embedding size.')
@click.option('--n_char_hidden', type=int, default=30,
              help='character hidden vector size.')
@click.option('--dropout_rate', type=float, default=0.33)
@click.option('--gpu', type=int, default=-1,
              help='gpu ID. when gpu=-1 use CPU mode.')
@click.option('--word_emb_file', type=click.Path())
@click.option('--vocab_file', type=click.Path())
@click.option('--vocab_char_file', type=click.Path())
@click.option('--dev_file', type=click.Path())
@click.option('--test_file', type=click.Path())
@click.option('--model_filename', type=click.Path())
@click.option('--delimiter', type=str, default='\t',
              help='delimiter string')
@click.option('--save_name', type=str, default='',
              help='save_name')
def train(train_file, **args):
    # load input_file
    main.run(train_file, is_train=True, **args)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--save_dir', type=str, default='save_model_dir',
              help='trained model save dir')
@click.option('--model_name', type=str, default='bilstm-cnn-crf',
              help="select from [bilstm-cnn-crf, bilstm-cnn]")
@click.option('--batchsize', type=int, default=32)
@click.option('--max_iter', type=int, default=100)
@click.option('--optimizer', type=str, default='adam',
              help="select from [adam, adadelta, sgd, sgd_mom]")
@click.option('--init_lr', type=float, default=0.01)
@click.option('--weight_decay', type=float, default=0.0)
@click.option('--use_lr_decay', type=int, default=0)
@click.option('--use_crf', type=int, default=1, help='use CRF flag.')
@click.option('--n_layer', type=int, default=1)
@click.option('--n_hidden', type=int, default=200)
@click.option('--n_vocab_min_cnt', type=int, default=0,
              help='min count of vocab.')
@click.option('--n_word_emb', type=int, default=100,
              help='word embedding size.')
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
@click.option('--save_name', type=str, default='',
              help='save_name')
@click.option('--predicted_output', type=str, default='',
              help='predicted_output')
@click.option('--model_filename', type=click.Path())
def predict(input_file, **args):
    main.run(input_file, is_train=False, **args)


@cli.command()
@click.argument('gold_file', type=click.Path(exists=True))
@click.argument('predicted_file', type=click.Path(exists=True))
@click.option('--tag_type', type=str, default='BIOES', help='select from [BIO, BIOES]')
def eval(gold_file, predicted_file, **args):
    evaluate.run(gold_file, predicted_file, **args)
