import math
import numpy as np
import scipy.sparse
from sklearn.svm import LinearSVC

np.set_printoptions(precision=2) # Print only 2 digits

class SVM:
    """
    Class to perform Multi Class SVM Naive Bayes classification
    """
    def __init__(self, Data, featureExtractor):
        self.Data = Data
        self.featureExtractor = featureExtractor
        self.SVMPredictor = LinearSVC(C=1.0) # SVM predictor

        # Learn the parameters when you instantiate the class
        self.learn()

    def setNewFeatureExtractor(self, newFeatureExtractor):
        """
        Method to update feature extractor and learn.
        """
        self.featureExtractor = newFeatureExtractor
        self.learn()

    def learn(self, convert=True):
        """
        Learns a Multi Class SVM predictor based on the featureExtractor.
        """
        if convert:
            self.Data.convertDataToArray(self.featureExtractor)

        self.SVMPredictor.fit(self.Data.trainArray, self.Data.trainLabelArray)

    def predict(self, reviewArray):
        """
        Returns a vector of labels with a prediction for each row in the array
        """
        return self.SVMPredictor.predict(reviewArray)

    def getTrainingMisClass(self):
        """
        Return misclassification rate on training set.
        """
        predictedLabels = self.predict(self.Data.trainArray)
        labelDifference = np.subtract(predictedLabels, self.Data.trainLabelArray)
        return sum( 1.0 for difference in labelDifference if difference != 0 )/self.Data.numLines

    def getTestMisClass(self):
        """
        Return misclassification rate on test set.
        """
        predictedLabels = self.predict(self.Data.testArray)
        labelDifference = np.subtract(predictedLabels, self.Data.testLabelArray)
        return sum( 1.0 for difference in labelDifference if difference != 0 )/self.Data.testLines


    def getConfusionMatrix(self, asFraction = True):
        """
        Returns confusion matrix.
        Row: True Value
        Column: Prediction
        Entries: Counts
        asFraction: if False return counts, otherwise fraction
        """
        predictedLabels = self.predict(self.Data.testArray)
        confusionMatrix = np.zeros((5,5), np.int)
        for i, prediction in enumerate(predictedLabels):
            confusionMatrix[self.Data.testLabelArray[i] - 1, prediction - 1] += 1
        if asFraction:
            rowSums = confusionMatrix.sum(axis=1)
            return confusionMatrix.astype(float)/rowSums[:, np.newaxis]
        else:
            return confusionMatrix

    def getTestRMSE(self):
        """
        Returns a Root Mean Squared Error on test set.
        """
        MSETest = 0.0
        predictedLabels = self.predict(self.Data.testArray)
        for i, prediction in enumerate(predictedLabels):
            MSETest += (prediction - self.Data.testLabelArray[i])**2
        return math.sqrt( MSETest/self.Data.testLines )

    def getTrainingRMSE(self):
        """
        Returns a Root Mean Squared Error on training set.
        """
        MSETrain = 0.0
        predictedLabels = self.predict(self.Data.trainArray)
        for i, prediction in enumerate(predictedLabels):
            MSETrain += (prediction - self.Data.trainLabelArray[i])**2
        return math.sqrt( MSETrain/self.Data.numLines )

    def getInfo(self):
        """
        Prints info about model and various errors.
        """
        print "Using %s training reviews and %s test reviews" % (self.Data.numLines, self.Data.testLines)
        print "Number of features: %s" % self.Data.numFeatures
        print "Training Misclassification: %s" % self.getTrainingMisClass()
        print "Test Misclassification: %s" % self.getTestMisClass()
        print "Train RMSE: %s" % self.getTrainingRMSE()
        print "Test RMSE: %s" % self.getTestRMSE()
        print "Confusion Matrix: "
        print self.getConfusionMatrix()
        print "\n"
