import collections
import json
import math
import os.path
import sys
from utility import *

INTERCEPT = '-INTERCEPT-' # intercept token

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
    def __init__(self, filename, numLines=100, featureExtractor = extractWordFeatures, testLines=100):
        """
        Loads in json data from file at filename.
        """
        if not os.path.isfile(filename):
            raise RuntimeError, "The file '%s' does not exist" % filename

        self.filename = filename
        self.numLines = numLines
        self.testLines = testLines
        self.featureExtractor = extractWordFeatures
        self.data, self.testData = loadData(filename, numLines, testLines)
        self.weights = {}
        self.meanRating = None

    def letMeSeeThatData(self):
        print self.data

    def averageRating(self):
        """
        Calculates average star rating (from 1 to 5) of reviews in dataset.
        """
        self.meanRating = sum( float(review['stars']) for review in self.data )/self.numLines
        return self.meanRating

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
        # needs to run self.averageRating() first

        self.weights = {}
        #self.meanFeature = {}

        # mean-centering
        #for review in self.data:
        #    phi = self.featureExtractor(review['text'])
        #    increment(self.meanFeature, float(1/self.numLines), phi)

        numIters, eta = 10, 0.0005
        for t in range(numIters):
            for review in self.data:
                star = review['stars']
                text = review['text']
                phi = self.featureExtractor(text)
                phi[INTERCEPT] = 1
                #increment(phi, -1, self.meanFeature)
                coeff = dotProduct(self.weights, phi) - star + self.meanRating

                increment(self.weights, float(-eta*coeff), phi)

    def predictRating(self, review):
        """
        Predicts a star rating from 1 to 5 given the |review| text
        """
        phi = self.featureExtractor(review['text'])
        phi[INTERCEPT] = 1
#        increment(phi, -1, self.meanFeature)
        prediction = dotProduct(phi, self.weights) + self.meanRating
        if prediction <= 1:
            return 1
        elif prediction >= 5:
            return 5
        else:
            return prediction
            #return round(prediction)
