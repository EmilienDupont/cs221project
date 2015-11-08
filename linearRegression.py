from utility import *
import math

class LinearRegression:
    """
    Class to perform linear regression
    """

    def __init__(self, Data, featureExctractor, shuffle=False):
        self.Data = Data
        self.numLines = Data.numLines
        if(shuffle):
            self.shuffleData()

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

    def learn(self, verbose=False, numIters=10):
        """
        Learns a linear predictor based on the featureExtractor.
        Option to set learning rate |eta| and number of iterations
        |numIters|.
        """
        # setting eta as a function of the number of training examples and number of itreations
        # should prob be something more elaborate
        eta = 5.0/float(self.numLines * numIters)
        self.weights = {}

        # Extract all the features before
        AllFeatures = []
        for review in self.Data.trainData:
            text = review['text']
            AllFeatures.append(self.featureExtractor(text))

        for t in range(numIters):

            if verbose:
                print "Iteration:", t
                print "Training error: %s, test error: %s" % (self.getTrainingRMSE(), self.getTestRMSE())

            for index, review in enumerate(self.Data.trainData):
                star = review['stars']
                phi = AllFeatures[index]
                phi[self.INTERCEPT] = 1
                updateCoefficient = dotProduct(self.weights, phi) - star + self.Data.meanRating
                increment(self.weights, float(-eta*updateCoefficient), phi)

    def learnSlow(self, numIters=10, eta = 0.002, momentum=0.0, gamma=0.9):
        """
        Learns a linear predictor based on the featureExtractor.
        Option to set learning rate |eta| and number of iterations
        |numIters|.
        """

        def calculateUpdate(new, old, momentum):
            #Combines historical info with new info
            updateFactor = {}
            for key in new:
                new[key] = (1.0-momentum)*new[key]
            for key in old:
                if(abs(old[key]) > 0.00001):
                    #print old[key]
                    if(key in new):
                        new[key] += momentum*old[key]
                    else:
                        new[key] = momentum*old[key]
            return new

        self.weights = {}

        historicalUpdate = {}

        for t in range(numIters):
            for review in self.Data.trainData:
                star = review['stars']
                text = review['text']
                phi = self.featureExtractor(text)
                phi[self.INTERCEPT] = 1
                updateCoefficient = dotProduct(self.weights, phi) - star + self.Data.meanRating
                historicalUpdate = calculateUpdate(phi, historicalUpdate, momentum)
                increment(self.weights, float(-eta*updateCoefficient), historicalUpdate)
            eta *=gamma
            #weightSum = sum(self.weights.values())
            for key in self.weights:
                #self.weights[key] /= weightSum
                self.weights[key] *= (1.0-50*eta)


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

    def getInfo(self):
        """
        Prints info about model and various errors.
        """
        print "Using %s training reviews and %s test reviews" % (self.Data.numLines, self.Data.testLines)
        print "Number of features: %s" % len(self.weights)
        print "Highest weighted features:", max( (self.weights[weight], weight) for weight in self.weights )
        print "Training RMSE: %s" % self.getTrainingRMSE()
        print "Training Misclassification: %s" % self.getTrainingMisClass()
        print "Test RMSE: %s" % self.getTestRMSE()
        print "Test Misclassification: %s" % self.getTestMisClass()
        print "\n"

    def crossVal(self):
        #Prep Data
        trainNum = len(self.Data.trainData)
        testNum = len(self.Data.testData)
        folds = int(round(trainNum/testNum))
        allData = self.Data.trainData + self.Data.testData

        trainRMSE = []
        trainMC = []
        testRMSE = []
        testMC = []

        from random import shuffle
        shuffle(allData)

        for i in xrange(folds): #For every fold

            self.Data.trainData = []
            self.Data.testData = []

            for j in xrange(len(allData)): #Create datasets
                if(j >= i*testNum and j < (i+1)*testNum+1):
                    self.Data.testData.append(allData[j])
                else:
                    self.Data.trainData.append(allData[j])
            #Perform learning and evaluation
            self.learn()
            trainRMSE.append(self.getTrainingRMSE())
            trainMC.append(self.getTrainingMisClass())
            testRMSE.append(self.getTestRMSE())
            testMC.append(self.getTestMisClass())

        print "Using %s training reviews and %s test reviews with %s fold cross validation" % (self.Data.numLines, self.Data.testLines, folds)
        print "Average Training RMSE: %s" % (sum(trainRMSE)/len(trainRMSE))
        print "Average Training Misclassification: %s" % (sum(trainMC)/len(trainMC))
        print "Average Test RMSE: %s" % (sum(testRMSE)/len(testRMSE))
        print "Average Test Misclassification: %s" % (sum(testMC)/len(testMC))
        print "\n"
