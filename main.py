import features

from data import *
from linearRegression import *
from multiClassSVM import *
from naiveBayes import *

# Analyzing command line arguments
if len(sys.argv) < 2:
  print 'Usage:'
  print '  python %s <JSON file>' % sys.argv[0]
  exit()

inputFile = sys.argv[1]

leafWords = ['s','es','ed','er','ly','ing']
cw = features.readCommonWords('common_words.txt')

# Import Data
reviews = Data(inputFile, numLines = 50000, testLines = 5000)
reviews.getInfo()
reviews.shuffle()


# Different feature extractors
f1 = features.posNegClusterFeatures("embeddings.p", "dictionary.p", 'NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt', 200)
f2 = features.wordFeatures
f3 = features.positiveNegativeCountsWithClause('NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt')
f4 = features.emotionCounts('NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt')
f5 = features.stemmedWordFeatures(leafWords)
f6 = features.clusterFeatures("embeddings.p", "dictionary.p", 200)
f7 = features.wordFeaturesWithNegation(cw, leafWords)

# SVM
SVMModel = SVM(reviews, [f1, f5])
SVMModel.getInfo()

# Naive Bayes
naiveBayesModel = NaiveBayes(reviews, [f3, f4, f5])
naiveBayesModel.getInfo()

# Linear Regression
linearModel = LinearRegression(reviews, [f7, f3, f1])
linearModel.getInfo()

