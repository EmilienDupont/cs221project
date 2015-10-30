import features

from training import *
from linearRegression import *

# Analyzing command line arguments
if len(sys.argv) < 2:
  print 'Usage:'
  print '  python %s <JSON file>' % sys.argv[0]
  exit()

inputFile = sys.argv[1]

reviews = Data(inputFile, numLines = 1000, testLines = 100)

print 'Average rating of %s reviews is %s' % (reviews.numLines, reviews.averageRating())
print 'Mode rating of %s reviews is %s' % (reviews.numLines, reviews.modeRating())

# Set up an actual model
linearModel = LinearRegression(reviews, features.wordFeatures)

# train error
MSETrain = 0
for review in reviews.trainData:
    MSETrain += (review['stars'] - linearModel.predictRating(review))**2

MSETrain /= reviews.numLines

# Misclassification
MisClass = sum( 1.0 for review in reviews.trainData if review['stars'] != round(linearModel.predictRating(review)) )/reviews.numLines
print "Training RMSE: %s" % math.sqrt(MSETrain)
print "Training MisClass: %s" % MisClass

# testing error
MSETest = 0
for review in reviews.testData:
    MSETest += (review['stars'] - linearModel.predictRating(review))**2

MSETest /= reviews.testLines

print "Test RMSE: %s" % math.sqrt(MSETest)
