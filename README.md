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

## How to train?
### train [Ma and Hovy (2016)](https://arxiv.org/abs/1603.01354) model
```
$ deep-crf train input_file.txt --delimiter ' ' --model_name bilstm-cnn-crf
```
### Deep BiLSTM-CNN-CRF model (three layers)
```
$ deep-crf train input_file.txt --delimiter ' ' --model_name bilstm-cnn-crf --n_layer 3
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
```
### Additional Feature Support
```
$ deep-crf train input_file_multi.txt --delimiter ' ' --model_name bilstm-cnn-crf −−input idx 0,1 −−output idx 2
```

### Multi-Task Learning Support
```
$ deep-crf train input_file_multi.txt --delimiter ' ' --model_name bilstm-cnn-crf −−input idx 0 −−output idx 1,2
```

## How to predict?
```
$ deep-crf predict input_raw_file.txt --model_name bilstm-cnn-crf --model_filename bilstm-cnn-crf_adam.model
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








