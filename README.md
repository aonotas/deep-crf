<p align="center"><img src="https://github.com/aonotas/deep-crf/blob/master/deep-crf.png" width="150"></p>

# DeepCRF: Neural Networks and CRFs for Sequence Labeling
A implementation of Conditional Random Fields (CRFs) with Deep Learning Method.

DeepCRF is a sequene labeling library that uses neural networks and CRFs in Python using Chainer, a flexible deep learning framework.

## How to install?
```
git clone https://github.com/aonotas/deep-crf.git
cd deep-crf
python setup.py install
```

Note that Chainer version is `1.2.40`
```
pip install 'chainer==1.24.0'
```

## How to train?
### train [Ma and Hovy (2016)](https://arxiv.org/abs/1603.01354) model
```
$ mkdir save_model_dir
$ deep-crf train input_file.txt --delimiter ' ' --model_name bilstm-cnn-crf
```
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
$ deep-crf train input_file.txt --delimiter ' ' --model_name bilstm-cnn-crf --n_layer 3
```

### Set Pretrained Word Embeddings 
```
$ deep-crf train input_file.txt --delimiter ' ' --model_name bilstm-cnn-crf --n_layer 3 --word_emb_file ./glove.6B.100d.txt
```
(Now only support Glove vecotr format. I will support word2vec format.)

### Additional Feature Support
```
$ deep-crf train input_file_multi.txt --delimiter ' ' --model_name bilstm-cnn-crf −−input idx 0,1 −−output idx 2
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

### Multi-Task Learning Support
```
$ deep-crf train input_file_multi.txt --delimiter ' ' --model_name bilstm-cnn-crf −−input idx 0 −−output idx 1,2
```

## How to predict?
```
$ deep-crf predict input_raw_file.txt --model_name bilstm-cnn-crf --model_filename bilstm-cnn-crf_adam_epoch10.model --predicted_output predicted.txt
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
CRFsuit                                                                    | 96.39
deep-crf                                                                   | 97.45
[dos Santos and Zadrozny (2014)](http://proceedings.mlr.press/v32/santos14.pdf) | 97.32
[Ma and Hovy (2016)](https://arxiv.org/abs/1603.01354)                     | 97.55  


### Named Entity Recognition (NER)
Model                                                                           | Prec. | Recall | F1
------------------------------------------------------------------------------- | :---: | :---:  | :---: 
CRFsuit                                                                         | 84.43 | 83.60  | 84.01
deep-crf                                                                        | 90.82 | 91.11  | 90.96
[Ma and Hovy (2016)](https://arxiv.org/abs/1603.01354)                          | 91.35 | 91.06  | 91.21


### Chunking
Model                                                                           | Prec. | Recall | F1
------------------------------------------------------------------------------- | :---: | :---:  | :---: 
CRFsuit                                                                         | 93.77 | 93.45  | 93.61
deep-crf                                                                        | 94.67 | 94.43  | 94.55
[Huang et al. (2015)](https://arxiv.org/abs/1508.01991)                         |   -   |   -    | 94.46








