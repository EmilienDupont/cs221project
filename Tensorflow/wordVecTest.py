
#from word2vec import *
from encodeWords import *
import tensorflow.python.platform
import collections
import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0, '../')
from data import *

# Analyzing command line arguments
if len(sys.argv) < 2:
  print ("Usage:")
  print ("  python %s <JSON file>" % sys.argv[0])
  exit()

inputFile = sys.argv[1]

# Import Data
reviews = Data(inputFile, numLines = 1000, testLines = 1000)
reviews.shuffle()

vocab = []

for review in reviews.trainData:
	for word in review['text'].split():

		vocab.append(word.encode("utf-8"))

embeddings = encodeWords(vocab, 1000, 128, 1)

#print tf.nn.embedding_lookup(embeddings, ['if', 'then'])