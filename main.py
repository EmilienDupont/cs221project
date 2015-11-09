import features

from data import *
from linearRegression import *
from naiveBayes import *

# Analyzing command line arguments
if len(sys.argv) < 2:
  print 'Usage:'
  print '  python %s <JSON file>' % sys.argv[0]
  exit()

inputFile = sys.argv[1]

leafWords = ['s','es','ed','er','ly','ing']
cw = features.readCommonWords('../data/common_words.txt')

# Import Data
reviews = Data(inputFile, numLines = 1000, testLines = 100)

# Without shuffling
"""
print "No shuffling"

# Set up Naive Bayes model
print "Naive Bayes"
print "Word features:"
naiveBayesModel = NaiveBayes(reviews, features.wordFeatures)
naiveBayesModel.getInfo()

print "With stemmed word features"
naiveBayesModel.setNewFeatureExtractor(features.stemmedWordFeatures(leafWords))
naiveBayesModel.getInfo()

print "Without most common words"
noCommonWords = features.wordFeaturesNoCommonWords(cw)
naiveBayesModel.setNewFeatureExtractor(noCommonWords)
naiveBayesModel.getInfo()

# Set up linear model
print "Linear model"
linearModel = LinearRegression(reviews, features.wordFeatures)
print '\n'
print "Using word features:"
linearModel.getInfo()

# Change featureExtractor
linearModel.setNewFeatureExtractor(features.stemmedWordFeatures(leafWords))
print "Using stemmed word features:"
linearModel.getInfo()

# Without common words
#print 'Please enter path to file with most common words: '
#cwFile = input()
#cw = features.readCommonWords(cwFile)
noCommonWords = features.wordFeaturesNoCommonWords(cw)
linearModel.setNewFeatureExtractor(noCommonWords)
linearModel.getInfo()
"""

# same things but with shuffling enabled
print "With Shuffling:"
reviews.shuffle()

# Set up Naive Bayes model
print "Naive Bayes"
print "Word features:"
naiveBayesModel = NaiveBayes(reviews, features.wordFeatures)
naiveBayesModel.getInfo()

print "With stemmed word features"
naiveBayesModel.setNewFeatureExtractor(features.stemmedWordFeatures(leafWords))
naiveBayesModel.getInfo()

print "Without most common words"
noCommonWords = features.wordFeaturesNoCommonWords(cw)
naiveBayesModel.setNewFeatureExtractor(noCommonWords)
naiveBayesModel.getInfo()

# Set up linear model
print "Linear model"
linearModel = LinearRegression(reviews, features.wordFeatures)
print '\n'
print "Using word features:"
linearModel.getInfo()

# Change featureExtractor
linearModel.setNewFeatureExtractor(features.stemmedWordFeatures(leafWords))
print "Using stemmed word features:"
linearModel.getInfo()

# Without common words
#print 'Please enter path to file with most common words: '
#cwFile = input()
#cw = features.readCommonWords(cwFile)
print "Without common words"
noCommonWords = features.wordFeaturesNoCommonWords(cw)
linearModel.setNewFeatureExtractor(noCommonWords)
linearModel.getInfo()
