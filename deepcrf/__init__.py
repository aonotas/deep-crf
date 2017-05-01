import click
import logging

import main


@click.group()
def cli():
    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


@cli.command()
@click.argument('train_file', type=click.Path(exists=True))
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
@click.option('--n_char_emb', type=int, default=50,
              help='character embedding size.')
@click.option('--dropout_rate', type=float, default=0.33)
@click.option('--gpu', type=int, default=-1,
              help='gpu ID. when gpu=-1 use CPU mode.')
@click.option('--word_emb_file', type=click.Path())
@click.option('--dev_file', type=click.Path())
@click.option('--test_file', type=click.Path())
def train(train_file, **args):
    # load input_file
    print args
    main.train(train_file, **args)
