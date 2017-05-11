<p align="center"><img src="https://github.com/aonotas/deep-crf/blob/master/deep-crf.png" width="150"></p>

# DeepCRF: Neural Networks and CRFs for Sequence Labeling
A implementation of Conditional Random Fields (CRFs) with Deep Learning Method.

DeepCRF is a sequene labeling library that uses neural networks and CRFs in Python using Chainer, a flexible deep learning framework.

## How to install?
```
pip install deep-crf # (it will work in the future)
```

## How to run?
```
wget http://deep-crf.com/trained_model/NER_trained_BiLSTM-CNN-CRF.model
# install deep-crf
pip install deep-crf
# run deep-crf
deep-crf --model ./NER_trained_BiLSTM-CNN-CRF.model --input input_raw_text.txt --output output_ner.txt
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

## Plan 
- First release : 2017-05-12


