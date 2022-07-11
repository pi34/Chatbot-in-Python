import numpy as np
import nltk
import os
import tflearn
import tensorflow as tf
import random
import json
import pickle
import re

data = []

# Text Preprocessing

# Creating Bag of Words Vector

words = []
labels = []
docsX = []
docsY = []

for x in data:

  for y in x['patterns']:
    pattern = pattern.lower()
    pattern=re.sub(r'[^(a-zA-Z)\s]', '',pattern)
    wrd = nltk.word_tokenize(pattern)
    words.extend(wrd)
    docsX.append(wrd)
    docsY.append(x['tags'])

  if x['tags'] not in labels:
    labels.append(x['tags'])
    
# Lancaster Stemming and Vectorisation

stm = nltk.LancasterStemmer()
words = [stm.stem(w.lower() for w in words)]
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docsX):

  bag = []
  wrd = [stm.stem(w) for w in doc]

  for w in words:
    if w in wrd:
      bag.append(1)
    else:
      bag.append(0)

  out_row = out_empty[:]
  out_row[labels.index(docsY[x])] = 1
  training.append(bag)
  output.append(out_row)  

training = np.array(training)
output = np.array(output)
