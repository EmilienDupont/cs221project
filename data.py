import collections
import json
import numpy as np
import math
import os.path
import scipy.sparse
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

        # For converting data into a numpy array
        self.trainArray = np.array([])
        self.trainLabelArray = np.array([])
        self.testArray = np.array([])
        self.testLabelArray = np.array([])
        self.featuresToIndex = {}
        self.numFeatures = 0

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

    def convertDataToArray(self, featureExtractor):
        """
        Convert the train and test data into a numpy array.
        Rows: examples
        Columns: features
        Stored as a sparse COO Matrix
        """
        self.featuresToIndex = {} # Links feature to index in numpy array
                                  # For example {"hello" : 0, "world" : 1}
        featureIndex = 0

        AllFeatures = []
        trainFeatureList = []
        trainLabelList = []
        AllTestFeatures = []
        testFeatureList = []
        testLabelList = []

        # Fill out labels and feature indices for Train Set
        for index, review in enumerate(self.trainData):
            text = review['text']
            AllFeatures.append(featureExtractor(text))
            trainLabelList.append(review['stars'])
            # Create a dictionary that links a particular feature to its index
            # in the numpy array
            for feature in AllFeatures[index]:
                if feature not in self.featuresToIndex:
                    self.featuresToIndex[feature] = featureIndex
                    featureIndex += 1

        # Fill out labels and feature indices for Test Set
        for index, review in enumerate(self.testData):
            text = review['text']
            AllTestFeatures.append(featureExtractor(text))
            testLabelList.append(review['stars'])
            # Add features that did not appear in training set
            for feature in AllTestFeatures[index]:
                if feature not in self.featuresToIndex:
                    self.featuresToIndex[feature] = featureIndex
                    featureIndex += 1

        self.trainLabelArray = np.array(trainLabelList)
        self.testLabelArray = np.array(testLabelList)
        self.numFeatures = len(self.featuresToIndex)

        # Array will have an example on each row
        # The columns will correspond to each feature
        rowTrain = []; colTrain = []; entriesTrain = []
        rowTest = []; colTest = []; entriesTest = []

        # Fill out the train array
        for index, review in enumerate(AllFeatures):
            for feature in review:
                rowTrain.append(index)
                colTrain.append(self.featuresToIndex[feature])
                entriesTrain.append(review[feature])

        self.trainArray = scipy.sparse.coo_matrix((entriesTrain, (rowTrain, colTrain)), (self.numLines, self.numFeatures), dtype = np.float)

        # Fill out the test array
        for index, review in enumerate(AllTestFeatures):
            for feature in review:
                rowTest.append(index)
                colTest.append(self.featuresToIndex[feature])
                entriesTest.append(review[feature])

        self.testArray = scipy.sparse.coo_matrix((entriesTest, (rowTest, colTest)), (self.testLines, self.numFeatures), dtype = np.float)
