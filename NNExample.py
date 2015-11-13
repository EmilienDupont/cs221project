import features

from NeuralNet import *
from training import *
from linearRegression import *

# Analyzing command line arguments
if len(sys.argv) < 2:
  print 'Usage:'
  print '  python %s <JSON file>' % sys.argv[0]
  exit()

inputFile = sys.argv[1]

# Import Data
reviews = Data(inputFile, numLines = 100, testLines = 10)

# Try a neural net instead of a simple average of the 
#IN PROGRESS
NN = NeuralNet(reviews)
#superCoolNet.loadWeights(linearModel)
NN.SGD()
NN.test()
#superCoolNet.gradientCheck()

#result =  superCoolNet.predict("neural nets are the bomb more words are needed here words eight long")

#superCoolNet.getInfo()
#print result