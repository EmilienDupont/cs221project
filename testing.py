import json
import os.path
import collections

class Testing:
    """
    Class to test models.
    """
    def __init__(self, Train):
        """
        Loads in model and test data
        """
        self.testData = Train.testData
        self.fixedPrediction = 0

    def setFixedPrediction(self, val):
        self.fixedPrediction = val

    def letMeSeeThatData(self):
        print self.data

    def fixedPredictor(self):
        return self.fixedPrediction

    def evaluatePredictor(self):
        totalPoints = 0
        numCorrect = 0

        for review in self.data:
            prediction = self.fixedPredictor()
            truth = review['stars']
            if prediction == truth:
                numCorrect += 1
            totalPoints += 1
        return float(numCorrect)/float(totalPoints)
