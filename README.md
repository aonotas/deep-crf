<p align="center"><img src="https://github.com/aonotas/deep-crf/blob/master/deep-crf.png" width="150"></p>

# DeepCRF: Neural Networks and CRFs for Sequence Labeling
A implementation of Conditional Random Fields (CRFs) with Deep Learning Method.

DeepCRF is a sequence labeling library that uses neural networks and CRFs in Python using Chainer, a flexible deep learning framework.

## Supported version of Python
* Python 2.x
* Python 3.x

## Which version of Chainer is supported?
* Chainer v1.24.0
* Chainer v2.1.0

## How to install?
```
git clone https://github.com/aonotas/deep-crf.git
cd deep-crf
python setup.py install

# if you want to use Chainer v1.24.0
pip install 'chainer==1.24.0'

# if you want to use Chainer v2.1.0
pip install 'chainer==2.1.0'
pip install cupy # if you want to use CUDA
```

## How to train?
### train [Ma and Hovy (2016)](https://arxiv.org/abs/1603.01354) model
```
$ deep-crf train input_file.txt --delimiter=' ' --dev_file input_file_dev.txt --save_dir save_model_dir --save_name bilstm-cnn-crf_adam --optimizer adam
```

Note that `--dev_file` means path of development file to use early stopping.

```
$ cat input_file.txt
Barack  B−PERSON 
Hussein I−PERSON 
Obama   E−PERSON
is      O 
a       O 
man     O 
.       O

Yuji   B−PERSON 
Matsumoto E−PERSON 
is     O 
a      O 
man    O 
.      O
```
Each line is `word` and `gold tag`.
One line is represented by `word` `[ ](space)` `gold tag`.
Note that you should put `empty line (\n)` between sentences.
This format is called CoNLL format.


### Deep BiLSTM-CNN-CRF model (three layers)
```
$ deep-crf train input_file.txt --delimiter=' ' --n_layer 3  --dev_file input_file_dev.txt --save_dir save_model_dir --save_name bilstm-cnn-crf_adam --optimizer adam
```

### set Pretrained Word Embeddings 
```
$ deep-crf train input_file.txt --delimiter=' ' --n_layer 3 --word_emb_file ./glove.6B.100d.txt --word_emb_vocab_type replace_all --dev_file input_file_dev.txt
```

We prepare some vocab mode.
- `--word_emb_vocab_type`: select from [replace_all, replace_only, additional]
- `replace_all` : Replace training vocab by Glove embeddings's vocab.
- `replace_only` : Replace word embedding exists in training vocab.
- `additional` : Concatenate training vocab and Glove embeddings's vocab.

If you want to use word2vec embeddings, please convert Glove format.
```
$ head glove.6B.100d.txt
the -0.038194 -0.24487 0.72812 -0.39961 0.083172
dog -0.10767 0.11053 0.59812 -0.54361 0.67396
cat -0.33979 0.20941 0.46348 -0.64792 -0.38377
of -0.1529 -0.24279 0.89837 0.16996 0.53516
to -0.1897 0.050024 0.19084 -0.049184 -0.089737
and -0.071953 0.23127 0.023731 -0.50638 0.33923
in 0.085703 -0.22201 0.16569 0.13373 0.38239
```

### Additional Feature Support
```
$ deep-crf train input_file_multi.txt --delimiter=' ' --input_idx 0,1 --output_idx 2 --dev_file input_file_dev.txt --save_dir save_model_dir --save_name bilstm-cnn-crf_adam_additional --optimizer adam
```

```
$ cat input_file_multi.txt
Barack  NN B−PERSON 
Hussein NN I−PERSON 
Obama   NN E−PERSON
is      VBZ O 
a       DT  O 
man     NN  O 
.       .   O

Yuji  NN B−PERSON 
Matsumoto NN E−PERSON 
is      VBZ O 
a       DT  O 
man     NN  O 
.       .   O
```
Note that `--input_idx` means that input features (but word feature must be 0-index) like this example.


### Multi-Task Learning Support
(Now developing this multi-task learning mode...)
```
$ deep-crf train input_file_multi.txt --delimiter ' ' --model_name bilstm-cnn-crf --input idx 0 --output idx 1,2 
```

## How to predict? 
```
$ deep-crf predict input_raw_file.txt --delimiter=' ' --model_filename ./save_model_dir/bilstm-cnn-crf_adam_epoch3.model --save_dir save_model_dir --save_name bilstm-cnn-crf_adam  --predicted_output predicted.txt
```

Please use following format when `predict`.
```
$ cat input_raw_file.txt
Barack Hussein Obama is a man .
Yuji Matsumoto is a man .
```

Note that `--model_filename` means saved model file path.
Please set same `--save_name` in training step. 



## How to predict? (Additional Feature)
```
$ deep-crf predict input_file_multi.txt --delimiter=' ' --input_idx 0,1 --output_idx 2 --model_filename ./save_model_dir/bilstm-cnn-crf_multi_epoch3.model --save_dir save_model_dir --save_name bilstm-cnn-crf_multi  --predicted_output predicted.txt
```
Note that you must prepare CoNLL format input file when you use additional feature mode in training step.
```
$ cat input_file_multi.txt
Barack  NN B−PERSON 
Hussein NN I−PERSON 
Obama   NN E−PERSON
is      VBZ O 
a       DT  O 
man     NN  O 
.       .   O

Yuji  NN B−PERSON 
Matsumoto NN E−PERSON 
is      VBZ O 
a       DT  O 
man     NN  O 
.       .   O
```

## How to evaluate?
```
$ deep-crf eval gold.txt predicted.txt
$ head gold.txt
O
O
B-LOC
O
O

B-PERSON
```


## How to update?
```
cd deep-crf
git pull
python setup.py install
```

## Help (how to use)
```
deep-crf train --help
```

## If CUDNN ERROR
if you got CUDNN ERROR, please let me know in issues.

You can cudnn-off mode with `--use_cudnn=0`

## Features
DeepCRF provides following features.
- Bi-LSTM / Bi-GRU / Bi-RNN
- CNN for character-level representation
- Pre-trained word embedding
- Pre-trained character embedding
- CRFs at output layer
- CoNLL format input/output
- Raw text data input/output
- Training : Your variable files
- Test : Raw text file at command-line
- Evaluation : F-measure, Accuracy

## Experiment

### POS Tagging
Model                                                                      | Accuracy 
-------------------------------------------------------------------------- | :---: 
CRFsuite                                                                    | 96.39
deep-crf                                                                   | 97.45
[dos Santos and Zadrozny (2014)](http://proceedings.mlr.press/v32/santos14.pdf) | 97.32
[Ma and Hovy (2016)](https://arxiv.org/abs/1603.01354)                     | 97.55  


### Named Entity Recognition (NER)
Model                                                                           | Prec. | Recall | F1
------------------------------------------------------------------------------- | :---: | :---:  | :---: 
CRFsuite                                                                         | 84.43 | 83.60  | 84.01
deep-crf                                                                        | 90.82 | 91.11  | 90.96
[Ma and Hovy (2016)](https://arxiv.org/abs/1603.01354)                          | 91.35 | 91.06  | 91.21


### Chunking
Model                                                                           | Prec. | Recall | F1
------------------------------------------------------------------------------- | :---: | :---:  | :---: 
CRFsuite                                                                         | 93.77 | 93.45  | 93.61
deep-crf                                                                        | 94.67 | 94.43  | 94.55
[Huang et al. (2015)](https://arxiv.org/abs/1508.01991)                         |   -   |   -    | 94.46








