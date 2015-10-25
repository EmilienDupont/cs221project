from training import *

# Analyzing command line arguments
if len(sys.argv) < 2:
  print 'Usage:'
  print '  python %s <JSON file>' % sys.argv[0]
  exit()

inputFile = sys.argv[1]

train = Training(inputFile, 10000)

print 'Average rating of %s reviews is %s' % (train.numLines, train.averageRating())
print 'Mode rating of %s reviews is %s' % (train.numLines, train.modeRating())

train.learnPredictor()

# train error
MSE = 0
for review in train.data:
    MSE += (review['stars'] - train.predictRating(review))**2

MSE /= train.numLines
    
print "Training RMSE: %s" % math.sqrt(MSE)

# testing error
