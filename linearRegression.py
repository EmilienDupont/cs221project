from utility import *
import math
import matplotlib.pyplot as plt
import numpy as np


class LinearRegression:
    """
    Class to perform linear regression. Initialize with an instance of data class
    and a LIST of featureExtractors
    """

    def __init__(self, Data, featureExtractor):
        self.Data = Data
        self.numLines = Data.numLines
        self.featureExtractor = featureExtractor
        self.weights = {}
        self.INTERCEPT = '-INTERCEPT-' # intercept token
        self.allFeatures = []

        # Extract all features
        self.extractAllFeatures()

        # Learn the parameters when you instantiate the class
        self.learn()

    def setNewFeatureExtractor(self, newFeatureExtractor):
        """
        Method to update feature extractor list and learn.
        """
        self.featureExtractor = newFeatureExtractor
        self.extractAllFeatures()
        self.learn()

    def extractFeatures(self, text):
        """
        Extracts features using the list of feature exctractors for a single example |text|
        """
        features = {}
        for extractor in self.featureExtractor:
            features.update(extractor(text))
        return features

    def extractAllFeatures(self):
        """
        Method to extract features of all the training data.
        """
        self.allFeatures = []
        for review in self.Data.trainData:
            text = review['text']
            self.allFeatures.append(self.extractFeatures(text))
        print "Extracted features!"


    def learn(self, verbose=False, numIters=50, epsilon=0.1):
        """
        Learns a linear predictor based on the featureExtractor.
        Option to set learning rate |eta| and number of iterations
        |numIters|.
        """
        originalEta = 100.0/float(self.numLines * numIters)
        self.weights = {}
        oldObjective = self.getObjective()

        for t in range(numIters):

            eta = originalEta/math.sqrt(t+1)

            if verbose:
                print "Iteration:", t
                print "Training error: %s, test error: %s" % (self.getTrainingRMSE(), self.getTestRMSE())

            for index, review in enumerate(self.Data.trainData):
                star = review['stars']
                phi = self.allFeatures[index]
                phi[self.INTERCEPT] = 1
                updateCoefficient = dotProduct(self.weights, phi) - star
                increment(self.weights, float(-eta*updateCoefficient), phi)

            # Check for convergence
            newObjective = self.getObjective()
            difference = abs(oldObjective - newObjective)
            print "Iteration: %s, difference: %s" % (t, difference)
            if difference < epsilon:
                print "Converged!"
                break
            oldObjective = newObjective

    def getObjective(self):
        """
        Function to return current objective value.
        """
        objective = 0
        for index, review in enumerate(self.Data.trainData):
            objective += (review['stars'] - dotProduct(self.allFeatures[index], self.weights))**2
        return objective/self.Data.numLines

    def predictRating(self, review, verbose=False):
        """
        Predicts a star rating from 1 to 5 given the |review| text
        """
        phi = self.extractFeatures(review['text'])
        phi[self.INTERCEPT] = 1
        prediction = dotProduct(phi, self.weights)
        if verbose: print prediction
        if prediction <= 1:
            return 1
        elif prediction >= 5:
            return 5
        else:
            return prediction

    def getTrainingRMSE(self):
        """
        Returns a Root Mean Squared Error on training set.
        """
        MSETrain = 0
        for review in self.Data.trainData:
            MSETrain += (review['stars'] - self.predictRating(review))**2
        return math.sqrt( MSETrain/self.Data.numLines )

    def getTrainingMisClass(self):
        """
        Return misclassification rate on training set.
        Note that the predicted rating gets rounded: 4.1 -> 4, 4.6 -> 5
        """
        return sum( 1.0 for review in self.Data.trainData
                if review['stars'] != round(self.predictRating(review)) )/self.Data.numLines

    def getTestRMSE(self):
        """
        Returns a Root Mean Squared Error on test set.
        """
        MSETest = 0
        for review in self.Data.testData:
            MSETest += (review['stars'] - self.predictRating(review))**2
        return math.sqrt( MSETest/self.Data.testLines )

    def getTestMisClass(self):
        """
        Return misclassification rate on test set.
        Note that the predicted rating gets rounded: 4.1 -> 4, 4.6 -> 5
        """
        return sum( 1.0 for review in self.Data.testData
                if review['stars'] != round(self.predictRating(review)) )/self.Data.testLines

    def plotPredictedRatingHistograms(self, numBins=25):
        """
        Plots a histogram of the distribution of the predicted ratings on the test set
        """
        predictedRatings = [[],[],[],[],[]]
        # Fill out list of lists of predicted ratings. predictedRatings[0] contains
        # all the true 1 star reviews etc...
        for review in self.Data.testData:
            predictedRatings[review['stars'] - 1].append(self.predictRating(review))
        print "Plotting..."
        for i in range(5):
            plt.subplot(3,2,i+1)
            plt.hist(np.array(predictedRatings[i]),numBins)
        plt.show()
        print "Done plotting!"

    def getConfusionMatrix(self, asFraction = True):
        """
        Returns confusion matrix.
        Row: True Value
        Column: Prediction
        Entries: Counts
        asFraction: if False return counts, otherwise fraction
        """
        predictedLabels = []
        trueLabels = []
        for review in self.Data.testData:
            trueLabels.append(review['stars'])
            predictedLabels.append(round(self.predictRating(review)))
        confusionMatrix = np.zeros((5,5), np.int)
        for i, prediction in enumerate(predictedLabels):
            confusionMatrix[trueLabels[i] - 1, prediction - 1] += 1
        if asFraction:
            rowSums = confusionMatrix.sum(axis=1)
            return confusionMatrix.astype(float)/rowSums[:, np.newaxis]
        else:
            return confusionMatrix

    def getFeatureVecExample(self, numExample):
        """
        Get feature vector of an example
        """
        return self.allFeatures[numExample]

    def getInfo(self):
        """
        Prints info about model and various errors.
        """
        print "Using %s training reviews and %s test reviews" % (self.Data.numLines, self.Data.testLines)
        print "Number of features: %s" % len(self.weights)
        print "Highest weighted features:", max( (self.weights[word], word) for word in self.weights \
                                                if word != self.INTERCEPT)
        print "Training RMSE: %s" % self.getTrainingRMSE()
        print "Training Misclassification: %s" % self.getTrainingMisClass()
        print "Test RMSE: %s" % self.getTestRMSE()
        print "Test Misclassification: %s" % self.getTestMisClass()
        print "Confusion Matrix: "
        print self.getConfusionMatrix()
        print "\n"
