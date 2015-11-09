import collections
import json
import math
import os.path
import sys
from random import shuffle
from utility import *

class Data:
    """
    Class to load and handle data.
    """
    def __init__(self, filename, numLines=100, testLines=100):
        """
        Loads in json data from file at filename.
        """
        if not os.path.isfile(filename):
            raise RuntimeError, "The file '%s' does not exist" % filename

        self.filename = filename
        self.numLines = numLines
        self.testLines = testLines
        self.trainData, self.testData = loadData(filename, numLines, testLines)
        self.meanRating = None
        self.averageRating()

    def shuffle(self):
        """
        Method to shuffle train and test data
        """
        allData = self.trainData + self.testData
        shuffle(allData)
        self.trainData = allData[0:self.numLines]
        self.testData = allData[self.numLines:]

    def averageRating(self):
        """
        Calculates average star rating (from 1 to 5) of reviews in dataset.
        """
        self.meanRating = sum( float(review['stars']) for review in self.trainData )/self.numLines
        return self.meanRating

    def ratingDistribution(self):
        """
        Calculates most common rating of reviews
        """
        starDistribution = collections.Counter()
        for review in self.trainData:
            star = review['stars']
            starDistribution.update({star:1})
        return starDistribution

    def getInfo(self):
        """
        Prints info about data
        """
        print "%s training reviews and %s test reviews" % (self.numLines, self.testLines)
        print "Average rating: ", self.averageRating()
        distribution = self.ratingDistribution()
        for rating in distribution:
            print "%s reviews with %s star(s)" % (distribution[rating], rating)
