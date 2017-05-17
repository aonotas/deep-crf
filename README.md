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
```
deep-crf train input_file.txt --delimiter ' ' --model_name bilstm-cnn-crf
```


## How to predict?
```
deep-crf predict input_raw_file.txt --model_name bilstm-cnn-crf --model_filename bilstm-cnn-crf_adam.model
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

