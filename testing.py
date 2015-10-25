import json
import os.path
import collections

class Testing:
    """
    Class to load and test data.
    """
    def __init__(self, filename, startLine = 101, numLines=100):
        """
        Loads in json data from file at filename.
        """
        if not os.path.isfile(filename):
            raise RuntimeError, "The file '%s' does not exist" % filename

        self.numLines = numLines
        self.startLine = startLine
        self.data = []
        self.fixedPrediction = 0

        lineNum = 0
        with open(filename) as f:
            for line in f:
                if lineNum >= self.startLine and lineNum < self.numLines + self.startLine:
                    self.data.append(json.loads(line))
                lineNum += 1
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

