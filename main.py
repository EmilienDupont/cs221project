from training import *

# Analyzing command line arguments
if len(sys.argv) < 2:
  print 'Usage:'
  print '  python %s <JSON file>' % sys.argv[0]
  exit()

inputFile = sys.argv[1]

train = Training(inputFile, numLines = 10000, testLines = 10000)

print 'Average rating of %s reviews is %s' % (train.numLines, train.averageRating())
print 'Mode rating of %s reviews is %s' % (train.numLines, train.modeRating())

train.learnPredictor()

# train error
MSETrain = 0
for review in train.data:
    MSETrain += (review['stars'] - train.predictRating(review))**2

MSETrain /= train.numLines
    
print "Training RMSE: %s" % math.sqrt(MSETrain)

# testing error
MSETest = 0
for review in train.testData:
    MSETest += (review['stars'] - train.predictRating(review))**2
    
MSETest /= train.testLines

print "Test RMSE: %s" % math.sqrt(MSETest)