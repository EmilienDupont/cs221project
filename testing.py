import json
import os.path
import collections

class Testing:
    """
    Class to load and test data.
    """
    def __init__(self, Train, extraLines=100):
        """
        Loads in json test data from same dataset as the training instance.
        """
        if not os.path.isfile(Train.filename):
            raise RuntimeError, "The file '%s' does not exist" % Train.filename

        self.extraLines = extraLines
        self.startLine = Train.numLines + 1
        self.testData = []
        self.fixedPrediction = 0

        lineNum = 0
        with open(Train.filename) as f:
            for line in f:
                if lineNum >= self.startLine and lineNum < self.startLine + self.extraLines:
                    self.data.append(json.loads(line))
                lineNum += 1

#### SHOULD CHANGE THIS TO HAVE A SEPARATE LOAD FUNCTION THAT LOADS EVERYTHING AND STORES self.TrainData, self.TestData

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

test = Testing('../../../../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json',10001,10000)
test.setFixedPrediction(5)

print 'Accuracy of %s reviews is %s' % (test.numLines, test.evaluatePredictor())
