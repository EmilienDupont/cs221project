#from word2vec import *
from encodeWords import *
import tensorflow.python.platform
import collections
import numpy as np
import tensorflow as tf
import re
import sys
sys.path.insert(0, '../')
from data import *
from features import tbl, negation_patterns, neg_match, neg_match, punct_patterns, removePunctuation, punct_mark

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

final_embeddings, reverse_dictionary = encodeWords(vocab, 1000, 128, 1)

print final_embeddings
print reverse_dictionary

def wordsWithNegation(leafWords):
  def stem(word):
    for leaf in leafWords:
      if word[-len(leaf):] == leaf:
	return word[:-len(leaf)]
      else:
	return word
      
  def extractor(text):
    processedWords = []
    prevNeg = False
    for word in text.split():
      wordStripped = stem(removePunctuation(word))
      if not prevNeg:
	prevNeg = (neg_match(word) != None)
	processedWords.append(wordStripped)
      else:
	negWord = wordStripped + "_NEG"
	processedWords.append(negWord)
      if punct_mark(word):
	prevNeg = False
    return processedWords
  
  return extractor

# dummy test
test = unicode("I was not pleased with the service, but i was pleased with the food.")
yolo = wordsWithNegation(['s','es','ed','er','ly','ing'])
swag = yolo(test)

print swag