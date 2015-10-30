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
reviews = Data(inputFile, numLines = 1000, testLines = 100)

print 'Average rating of %s reviews is %s' % (reviews.numLines, reviews.averageRating())
print 'Mode rating of %s reviews is %s' % (reviews.numLines, reviews.modeRating())

# Set up an actual model
linearModel = LinearRegression(reviews, features.wordFeatures)

print "Training RMSE: %s" % linearModel.getTrainingRMSE()
print "Training MisClass: %s" % linearModel.getTrainingMisClass()
print "Test RMSE: %s" % linearModel.getTestRMSE()
print "Test MisClass: %s" % linearModel.getTestMisClass()
