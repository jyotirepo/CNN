# -*- coding: utf-8 -*-
"""
Created on Sun May 23 00:01:37 2021

@author: jysethy
"""

from tensorflow.keras.preprocessing.text import one_hot


### sentences
sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]

#vocabulary size
voc_size = 10000


#one hot representation
onehot_repr=[one_hot(words,voc_size)for words in sent] 
print(onehot_repr)

#Word Embbedings representation

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import numpy as np

sent_length=8
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)

#dimension (number of feature for each word)
dim=10

model = Sequential()
model.add(Embedding(voc_size,10,input_length=sent_length))
model.compile('adam','mse')

model.summary()





