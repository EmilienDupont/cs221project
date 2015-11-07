from utility import *
import math
import numpy as np

from scipy import signal
from scipy import misc

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
    def __init__(self, filterLength = 5, numFilters = 10):
        self.filterLength = filterLength
        self.numFilters = numFilters
        self.Data = []
        self.nGramWeights = []
        
        weight_scale = 0.05
        bias_scale = 0.01;

        self.grads = {}
        self.checkGrads = {}
        self.step = {}
        self.model = {}
        self.model['weights1'] = weight_scale * np.random.randn(numFilters, 1, 1, filterLength)
        self.model['bias1'] = bias_scale * np.random.randn(numFilters)
        self.model['weights2'] = weight_scale * np.random.randn(numFilters, numFilters, 1, filterLength)
        self.model['bias2'] = bias_scale * np.random.randn(numFilters)
        
        self.conv_param = {'stride': 1, 'pad': 0}
        self.pool_param = {'pool_height': 1, 'pool_width': 1, 'stride_h': 1, 'stride_w': 1}

    def SGD(self, learningRate=0.001, decayFactor=0.95, momentum = 0.9, numIters=100):
        lr = learningRate

        for key in self.model:
            self.step[key] = 0

        for iter in range(0, numIters):
            print "Iteration: %d" %(iter)
            for review in self.Data.trainData:
                star = review['stars']
                text = review['text']
                if len(text.split()) > 5:
                    #print star, text
                    self.predict(text, star) #Calculate gradients on example
                    for key in self.model:
                        self.step[key] = self.step[key]*momentum + (1-momentum)*self.grads[key]
                        self.model[key] -= lr*self.step[key]
                        #print self.grads[key]
                    lr *= decayFactor
                        #print key

        print self.model
        #print self.grads[key]
        print "done"

        for review in self.Data.trainData:
            print review['text']
            print self.predict(review['text']), "truth: ", review['stars']

        #print self.Data.trainData[1]['text']
        #print self.predict(self.Data.trainData[1]['text']), "truth: ", self.Data.trainData[1]['stars']
        #print self.Data.trainData[99]['text']
        #print self.predict(self.Data.trainData[99]['text']), "truth: ", self.Data.trainData[99]['stars']

        print self.Data.testData[0]['text']
        print self.predict(self.Data.testData[0]['text']), "truth: ", self.Data.testData[0]['stars']
        #print self.Data.testData[9]['text']
        #print self.predict(self.Data.testData[9]['text']), "truth: ", self.Data.testData[9]['stars']

    def getInfo(self):
            """
            Prints info about model and various errors.
            """
            print "Using %s training reviews and %s test reviews" % (self.Data.numLines, self.Data.testLines)
            print "Training RMSE: %s" % self.getTrainingRMSE()
            print "Training Misclassification: %s" % self.getTrainingMisClass()
            print "Test RMSE: %s" % self.getTestRMSE()
            print "Test Misclassification: %s" % self.getTestMisClass()
            print "\n"

    def getTrainingMisClass(self):
        """
        Return misclassification rate on training set.
        Note that the predicted rating gets rounded: 4.1 -> 4, 4.6 -> 5
        """
        return sum( 1.0 for review in self.Data.trainData
                if review['stars'] != round(self.predict(review['text'])) )/self.Data.numLines

    def getTrainingRMSE(self):
        """
        Returns a Root Mean Squared Error on training set.
        """
        MSETrain = 0
        for review in self.Data.trainData:
            MSETrain += (review['stars'] - self.predict(review['text']))**2
        return math.sqrt( MSETrain/self.Data.numLines )


    def getTestMisClass(self):
        """
        Return misclassification rate on test set.
        Note that the predicted rating gets rounded: 4.1 -> 4, 4.6 -> 5
        """
        return sum( 1.0 for review in self.Data.testData
                if review['stars'] != round(self.predict(review['text'])) )/self.Data.testLines

    def getTestRMSE(self):
        """
        Returns a Root Mean Squared Error on test set.
        """
        MSETest = 0
        for review in self.Data.testData:
            MSETest += (review['stars'] - self.predict(review['text']))**2
        return math.sqrt( MSETest/self.Data.testLines )

    def getWordValueVector(self, text):
        valVec = []
        for word in text.split():
            if(word in self.nGramWeights):
                valVec.append(self.nGramWeights[word])
        
        #print valVec
        valMat = np.zeros([1,1,1,len(valVec)])
        for i in range(0, len(valVec)):
            valMat[0,0,0,i] = valVec[i]

        return valMat

    def loadWeights(self, unigramModel):
        self.nGramWeights = unigramModel.weights
        self.Data = unigramModel.Data

    def gradientCheck(self):
        review = self.Data.trainData[0]
        text = review['text']
        star = review['stars']
        origLoss = self.predict(text, star) #Calculate gradients on example
        originalGrads = self.grads
        for key in self.model:
            self.checkGrads[key] = 0*self.model[key]
            error = 0
            for (w,x,y,z), val in np.ndenumerate(self.model[key]):
                val += 0.01
                loss = self.predict(text, star)
                self.checkGrads[key][w,x,y,z] = (loss[0] - origLoss[0])/0.01
                val -= 0.01
        print originalGrads
        print self.checkGrads


    def Pass(self, valMat, truth):

        W1, b1, W2, b2 = self.model['weights1'], self.model['bias1'], self.model['weights2'], self.model['bias2']
        #W1, b1 = self.model['weights1'], self.model['bias1']
        N, C, H, W = valMat.shape

        #Forward Pass
        #Set1 
        #print "Forward Shapes"
        #print "Input ", valMat.shape, valMat
        a1, cache1 = self.Conv_Forward(valMat, W1, b1)
        #print self.model['weights1']
        #print "conv ", a1.shape, a1
        a2, cache2 = self.ReLu_Forward(a1)
        #print "ReLu ", a2.shape, a2
        a3, cache3 = self.Max_Pool_Forward(a2)
        #print "pool ", a3.shape, a3

        a4, cache4 = self.Conv_Forward(a3, W2, b2)
        #print self.model['weights1']
        #print "conv ", a1.shape, a1
        a5, cache5 = self.ReLu_Forward(a4)
        #print "ReLu ", a2.shape, a2
        a6, cache6 = self.Max_Pool_Forward(a5)
        #print "pool ", a3.shape, a3

        if(truth == None):
            return self.Mean(a6)

        #Backward Pass
        #print "Backward Shapes"
        loss, c1 = self.Mean_Loss(a6, truth)
        #print "after Loss ", c1.shape
        #print c1

        c5 = self.Max_Pool_Backward(c1, cache6)
        #print "after depool ", c2.shape
        #print c2
        c6 = self.ReLu_Backward(c5, cache5)
        #print "after deRelu ", c3.shape
        #print c3
        c7,  dW2, db2 = self.Conv_Backward(c6, cache4)

        c2 = self.Max_Pool_Backward(c7, cache3)
        #print "after depool ", c2.shape
        #print c2
        c3 = self.ReLu_Backward(c2, cache2)
        #print "after deRelu ", c3.shape
        #print c3
        c4,  dW1, db1 = self.Conv_Backward(c3, cache1)
        #print "after deconv ", c4.shape
        #print c4


        self.grads = {'weights1': dW1, 'bias1': db1, 'weights2': dW2, 'bias2': db2}

        return self.Mean(a6), self.grads

    def backprop(self, text):
            score, Inter1, inter2, inter3, Inter4 = forwardPass(text)

    def predict(self, text, truth=None):
        return self.Pass(self.getWordValueVector(text), truth)

    def ReLu_Forward(self, x):
        #out = np.maximum(0, x)
        out = np.where(x > 0, x, 0.001*x)
        cache = x
        return out, cache

    def ReLu_Backward(self, dout, cache):
        x = cache
        dx = np.where(x > 0, dout, 0.001*dout)
        return dx

    def Mean_Loss(self, x, y):
        loss = 0.5*(self.Mean(x) - y)**2
        dx = x*0.0 + (self.Mean(x) - y) #/(np.prod(x.shape))
        #return loss, (dx*0 + 1)
        return loss, dx

    def Mean(self, x):
        return np.mean(x)

    def Conv_Forward(self, x, w, b):
        """
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        Returns a tuple of:
        - out: Output data.
        - cache: (x, w, b, conv_param)
        """
        out = None
        pad = self.conv_param['pad']
        stride = self.conv_param['stride']
        H = x.shape[2]
        W = x.shape[3]
        HH = w.shape[2]
        WW = w.shape[3]
        N = x.shape[0]
        F = w.shape[0]

        out = np.zeros([N,F,H,W])

        for i in xrange(x.shape[0]): #For every data sample
            for j in xrange(x.shape[1]): #For every color
                temp = x[i,j,:,:] #store layer to make things simpler
                #temp = np.pad(temp, pad, 'constant')
                for f in xrange(w.shape[0]): #For every filter
                    filt = w[f,j,:,:]
                    out[i,f,:,:] += signal.correlate2d(temp, filt, mode='same', boundary='fill', fillvalue=0) + b[f]/x.shape[1]
        cache = (x, w, b, self.conv_param)
        return out, cache

    def Conv_Backward(self, dout, cache):
        """
        Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        [x, w, b, conv_param] = cache
        pad = self.conv_param['pad']
        stride = self.conv_param['stride']
        H = x.shape[2]
        W = x.shape[3]
        HH = w.shape[2]
        WW = w.shape[3]
        N = x.shape[0]
        F = w.shape[0]
        l = 0
        k = 0

        dw = 0*w
        dx = 0*x
        mask = np.ones([H - HH + 1, W - WW + 1])


        for i in xrange(x.shape[0]): #For every data sample
            for j in xrange(x.shape[1]): #For every color
                temp = dout[i,j,:,:] #store layer to make things simpler
                #temp = np.pad(temp, pad, 'constant')
                for f in xrange(w.shape[0]): #For every filter
                    filt = w[f,j,:,:]
                    dw[f,j,:,:] += signal.convolve2d(temp, mask, mode='valid')
                    dx[i,j,:,:] += signal.convolve2d(temp, filt, mode='same', boundary='fill', fillvalue=0)

        db = np.sum(dout, (0, 2, 3))

        return dx, dw, db

    def Max_Pool_Forward(self, x):
        """
        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
        - 'pool_height': The height of each pooling region
        - 'pool_width': The width of each pooling region
        - 'stride': The distance between adjacent pooling regions

        Returns a tuple of:
        - out: Output data
        - cache: (x, pool_param)
        """
        out = None

        [N,C,H,W ] = x.shape
        ph = self.pool_param['pool_height']
        pw = self.pool_param['pool_width']
        sth = self.pool_param['stride_h']
        stw = self.pool_param['stride_w']
        out = np.zeros((N,C,H/ph, W/pw))
        mask = 0*x

        for i in xrange(N): #For each datapoint
            for j in xrange(C): #For each color
                k = 0
                while k < (out.shape[2]):
                    l = 0
                    while l < (out.shape[3]):
                        temp = x[i,j,k*sth:k*sth + ph,l*stw:l*stw + pw]
                        a = np.argmax(temp)
                        b = np.unravel_index(a, temp.shape)
                        out[i,j,k,l] = temp[b]
                        mask[i,j,b[0]+k*sth,b[1]+l*stw] = 1
                        l += 1
                    k += 1
        cache = mask
        return out, cache

    def Max_Pool_Backward(self, dout, cache):
        """
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.

        Returns:
        - dx: Gradient with respect to x
        """
        [N,C,H,W ] = dout.shape
        mask = cache
        [NN,CC,HH,WW] = mask.shape
        ph = self.pool_param['pool_height']
        pw = self.pool_param['pool_width']
        sth = self.pool_param['stride_h']
        stw = self.pool_param['stride_w']

        dx = mask


        for i in xrange(NN):
            for j in xrange(CC):
                n = 0
                m = 0
                for k in xrange(HH):
                    for l in xrange(WW):
                        if(mask[i,j,k,l] == 1):
                            dx[i,j,k,l] = dout[i,j,n,m]
                            n += 1
                            if(n == H):
                                n = 0
                                m += 1
        return dx




