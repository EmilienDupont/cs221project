from utility import *
import math
import numpy as np

class CNN:
    """
    Class to perform deep learned neural network big data platform disrupts
    Structure:
        Conv
        Relu
        Pool
        Conv
        Relu 
        Pool
     """
    def __init__(self, filterLength = 5, numFilters = 3):
        self.filterLength = filterLength
        self.numFilters = numFilters
        self.Data = []
        self.nGramWeights = []
        self.weights1 = np.zeros((numFilters,filterLength)) #filterLength x numFilters
        self.weights2 = np.zeros((numFilters,filterLength)) #filterLength x numFilters

    def SGD(self, learningRate, decayFactor, numIters):
        for t in range(numIters):
            pass

    def getWordValueVector(self, text):
        valVec = []
        for word in text:
            valVec.append(self.nGramWeights[word])
        return valVec

    def loadWeights(self, unigramModel):
        self.weights = unigramModel.weights
        self.data = unigramModel.Data

    def forwardPass(valVec):
        Inter1 = np.zeros((self.numFilters,len(valVec)))
        #Conv
        for i in range(0, self.numFilters):
            Inter1[i,:] = np.convolve(valVec, self.weights1, 'same')
        #ReLu
        for n in np.nditer(Inter1):
            n = max(n, 0.0)
        #pool
        Inter2 = amax(Inter1, axis=0)

        Inter3 = np.zeros((self.numFilters,len(valVec)))
        #Conv
        for i in range(0, self.numFilters):
            Inter1[i,:] = np.convolve(valVec, self.weights1, 'same')
        #ReLu
        for n in np.nditer(Inter1):
            n = max(n, 0.0)
        #pool
        InterMax = 0*Inter3
        #InterMax(np.where(Inter3 ==)) = 
        Inter4 = amax(Inter1, axis=0)
        return (np.mean(Inter4), Inter1, inter2, inter3, inter4, )

        def backprop():
            pass

        def predict(text):
            return forwardPass(getWordValueVector(text))[0]