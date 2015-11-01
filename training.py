import collections
import json
import math
import os.path
import sys
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

    def letMeSeeThatData(self):
        print self.trainData

    def averageRating(self):
        """
        Calculates average star rating (from 1 to 5) of reviews in dataset.
        """
        self.meanRating = sum( float(review['stars']) for review in self.trainData )/self.numLines
        return self.meanRating

    def modeRating(self):
        """
        Calculates most common rating of reviews
        """
        C = collections.Counter()
        for review in self.trainData:
            s = review['stars']
            C.update({s:1})
        return C.most_common(1)[0][0]
