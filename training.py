import collections
import json
import os.path
import sys
from utility import *

# Analysing command line arguments
if len(sys.argv) < 2:
  print 'Usage:'
  print '  python %s <JSON file>' % sys.argv[0]
  exit()

inputFile = sys.argv[1]

def extractWordFeatures(x):
    words, features = x.split(), {}
    for word in words:
        if (word in features):
            features[word] += 1
        else:
            features[word] = 1
    return features

class Training:
    """
    Class to load and train data.
    """
    def __init__(self, filename, numLines=100, featureExtractor = extractWordFeatures):
        """
        Loads in json data from file at filename.
        """
        if not os.path.isfile(filename):
            raise RuntimeError, "The file '%s' does not exist" % filename
        
        self.filename = filename
        self.numLines = numLines
        self.featureExtractor = extractWordFeatures
        self.data = []

        lineNum = 0
        with open(filename) as f:
            for line in f:
                if lineNum < self.numLines:
                    self.data.append(json.loads(line))
                    lineNum += 1

    def letMeSeeThatData(self):
        print self.data

    def averageRating(self):
        """
        Calculates average star rating (from 1 to 5) of reviews in dataset.
        """
        return sum( float(review['stars']) for review in self.data )/self.numLines

    def modeRating(self):
        """
        Calculates most common rating of reviews
        """
        C = collections.Counter()
        for review in self.data:
            s = review['stars']
            C.update({s:1})
        return C.most_common(1)[0][0]
    
    def learnPredictor(self):
        """
        Learns a linear predictor based on the featureExtractor
        """
        self.weights = {}
        
        numIters, eta = 20, 0.025
        for t in range(numIters):
            for review in self.data:
                star = review['stars']
                text = review['text']
                phi = self.featureExtractor(text)
                
                coeff = dotProduct(self.weights, phi)
                increment(self.weights, float(-eta), phi)
    
    def predictRating(self, review):
        """
        Predicts a star rating from 1 to 5 given the |review| text
        """
        phi = self.featureExtractor(review['text'])
        prediction = dotProduct(phi, self.weights)
        if prediction <= 1:
            return 1
        elif prediction >= 5:
            return 5
        else:
            return round(prediction)

#train = Training('../../../../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json',10000)
train = Training(inputFile, 1000)
#train.letMeSeeThatData()
print 'Average rating of %s reviews is %s' % (train.numLines, train.averageRating())

print 'Mode rating of %s reviews is %s' % (train.numLines, train.modeRating())

train.learnPredictor()