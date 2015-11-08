import features

from training import *
from linearRegression import *
from naiveBayes import *

# Analyzing command line arguments
if len(sys.argv) < 2:
  print 'Usage:'
  print '  python %s <JSON file>' % sys.argv[0]
  exit()

inputFile = sys.argv[1]

# Import Data
reviews = Data(inputFile, numLines = 5000, testLines = 500)
reviews.shuffle()

# Set up Naive Bayes model
print "Naive Bayes with word features"
naiveBayesModel = NaiveBayes(reviews, features.wordFeatures)
naiveBayesModel.getInfo()

print "Naive Bayes with stemmed word features"
leafWords = ['s','es','ed','er','ly','ing']
naiveBayesModel.setNewFeatureExtractor(features.stemmedWordFeatures(leafWords))
naiveBayesModel.getInfo()

# Set up linear model
print "Using linear model"
linearModel = LinearRegression(reviews, features.wordFeatures)
print '\n'
print "Using word features:"
linearModel.getInfo()

# Change featureExtractor
linearModel.setNewFeatureExtractor(features.stemmedWordFeatures(leafWords))
print "Using stemmed word features:"
linearModel.getInfo()

# Without common words
print 'Please enter path to file with most common words: '
cwFile = input()
cw = features.readCommonWords(cwFile)
noCommonWords = features.wordFeaturesNoCommonWords(cw)
linearModel.setNewFeatureExtractor(noCommonWords)
linearModel.getInfo()
