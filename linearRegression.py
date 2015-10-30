from utility import *
import math

class LinearRegression:
    """
    Class to perform linear regression
    """

    def __init__(self, Data, featureExctractor):
        self.Data = Data
        self.featureExtractor = featureExctractor
        self.weights = {}
        self.INTERCEPT = '-INTERCEPT-' # intercept token

        # Learn the parameters when you instantiate the class
        self.learn()

    def setNewFeatureExtractor(self, newFeatureExtractor):
        """
        Method to update feature extractor and learn.
        """
        self.featureExtractor = newFeatureExtractor
        self.learn()

    def learn(self, numIters=10, eta = 0.0005):
        """
        Learns a linear predictor based on the featureExtractor.
        Option to set learning rate |eta| and number of iterations
        |numIters|.
        """
        self.weights = {}

        for t in range(numIters):
            for review in self.Data.trainData:
                star = review['stars']
                text = review['text']
                phi = self.featureExtractor(text)
                phi[self.INTERCEPT] = 1
                updateCoefficient = dotProduct(self.weights, phi) - star + self.Data.meanRating
                increment(self.weights, float(-eta*updateCoefficient), phi)

    def predictRating(self, review):
        """
        Predicts a star rating from 1 to 5 given the |review| text
        """
        phi = self.featureExtractor(review['text'])
        phi[self.INTERCEPT] = 1
        prediction = dotProduct(phi, self.weights) + self.Data.meanRating
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


