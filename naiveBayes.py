import numpy as np
import scipy.sparse
from sklearn.naive_bayes import MultinomialNB

class NaiveBayes:
    """
    Class to perform Naive Bayes classification
    """
    def __init__(self, Data, featureExctractor):
        self.Data = Data
        self.featureExtractor = featureExctractor
        self.trainArray = np.array([])
        self.trainLabelArray = np.array([])
        self.testArray = np.array([])
        self.testLabelArray = np.array([])
        self.featuresToIndex = {}
        self.numFeatures = 0
        self.NBPredictor = MultinomialNB() # The Naive Bayes Predictor

        # Learn the parameters when you instantiate the class
        self.learn()

    def setNewFeatureExtractor(self, newFeatureExtractor):
        """
        Method to update feature extractor and learn.
        """
        self.featureExtractor = newFeatureExtractor
        self.learn()

    def convertDataToArray(self):
        """
        Convert the train and test data into a numpy array.
        Note that this feature should be moved to Data class.
        """
        self.featuresToIndex = {} # Links feature to index in numpy array
        featureIndex = 0

        AllFeatures = []
        trainFeatureList = []
        trainLabelList = []
        AllTestFeatures = []
        testFeatureList = []
        testLabelList = []

        # Fill out labels and feature indices for Train Set
        for index, review in enumerate(self.Data.trainData):
            text = review['text']
            AllFeatures.append(self.featureExtractor(text))
            trainLabelList.append(review['stars'])
            # Create a dictionary that links a particular feature to its index
            # in the numpy array
            for feature in AllFeatures[index]:
                if feature not in self.featuresToIndex:
                    self.featuresToIndex[feature] = featureIndex
                    featureIndex += 1

        # Fill out labels and feature indices for Test Set
        for index, review in enumerate(self.Data.testData):
            text = review['text']
            AllTestFeatures.append(self.featureExtractor(text))
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
        #self.trainArray = np.zeros((self.Data.numLines, self.numFeatures), dtype = np.int)
        #self.testArray = np.zeros((self.Data.testLines, self.numFeatures), dtype = np.int)
        rowTrain = []; colTrain = []; entriesTrain = []
        rowTest = []; colTest = []; entriesTest = []

        # Fill out the train array
        for index, review in enumerate(AllFeatures):
            for feature in review:
                rowTrain.append(index)
                colTrain.append(self.featuresToIndex[feature])
                entriesTrain.append(review[feature])
                #self.trainArray[index, self.featuresToIndex[feature]] = review[feature]

        self.trainArray = scipy.sparse.coo_matrix((entriesTrain, (rowTrain, colTrain)), (self.Data.numLines, self.numFeatures), dtype = np.int)

        # Fill out the test array
        for index, review in enumerate(AllTestFeatures):
            for feature in review:
                rowTest.append(index)
                colTest.append(self.featuresToIndex[feature])
                entriesTest.append(review[feature])
                #self.testArray[index, self.featuresToIndex[feature]] = review[feature]

        self.testArray = scipy.sparse.coo_matrix((entriesTest, (rowTest, colTest)), (self.Data.testLines, self.numFeatures), dtype = np.int)


    def learn(self, convert=True):
        """
        Learns a Multinomial Naive Bayes predictor based on the featureExtractor.
        """
        if convert:
            self.convertDataToArray()

        self.NBPredictor.fit(self.trainArray, self.trainLabelArray)

    def predict(self, reviewArray):
        """
        Returns a vector of labels with a prediction for each row in the array
        """
        return self.NBPredictor.predict(reviewArray)

    def getTrainingMisClass(self):
        """
        Return misclassification rate on training set.
        """
        predictedLabels = self.predict(self.trainArray)
        labelDifference = np.subtract(predictedLabels, self.trainLabelArray)
        return sum( 1.0 for difference in labelDifference if difference != 0 )/self.Data.numLines

    def getTestMisClass(self):
        """
        Return misclassification rate on test set.
        """
        predictedLabels = self.predict(self.testArray)
        labelDifference = np.subtract(predictedLabels, self.testLabelArray)
        return sum( 1.0 for difference in labelDifference if difference != 0 )/self.Data.testLines

    def getInfo(self):
        """
        Prints info about model and various errors.
        """
        print "Using %s training reviews and %s test reviews" % (self.Data.numLines, self.Data.testLines)
        print "Number of features: %s" % self.numFeatures
        print "Training Misclassification: %s" % self.getTrainingMisClass()
        print "Test Misclassification: %s" % self.getTestMisClass()
        print "\n"
