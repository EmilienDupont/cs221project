import features

from training import *
from linearRegression import *

# Analyzing command line arguments
if len(sys.argv) < 2:
  print 'Usage:'
  print '  python %s <JSON file>' % sys.argv[0]
  exit()

inputFile = sys.argv[1]

# Import Data
reviews = Data(inputFile, numLines = 10000, testLines = 100)

# Set up an actual model
linearModel = LinearRegression(reviews, features.wordFeatures)
print '\n'
print "Using word features:"
linearModel.getInfo()

# Change featureExtractor
leafWords = ['s','es','ed','er','ly','ing']
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
