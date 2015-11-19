from utility import *
import math
import numpy as np
from data import shuffle

from scipy import signal
from scipy import misc
import collections

class NeuralNet:
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
    def __init__(self, Data, filterLength = 3, numFilters = 7):
        self.filterLength = filterLength
        self.numFilters = numFilters
        self.Data = Data

        self.nGramWeights = []
        
        #Initialize weights randomly in different ranges
        self.weight_scale = 0.1
        self.bias_scale = 0.01

        #Have different learning rates for different weights
        self.mult = {}
        self.mult['weights1'] = 5
        self.mult['bias1'] = 1
        self.mult['weights2'] = 2
        self.mult['bias2'] = 1
        self.mult['weights3'] = 2
        self.mult['bias3'] = 1
        self.mult['words'] = 2

        #Have regularization
        self.reg = {}
        self.reg['weights1'] = 0.00001
        self.reg['bias1'] = 0.00000
        self.reg['weights2'] = 0.00001
        self.reg['bias2'] = 0.00000
        self.reg['weights3'] = 0.00001
        self.reg['bias3'] = 0.00000
        self.reg['words'] = 0.00001

        self.labelVec = []
        self.grads = {}
        self.checkGrads = {}
        self.step = {}
        self.model = {}
        self.model['weights1'] = self.weight_scale * np.random.randn(numFilters, 1, 1, filterLength)
        self.model['bias1'] = self.bias_scale * np.random.randn(numFilters)
        #self.model['weights2'] = self.weight_scale * np.random.randn(numFilters, numFilters, 1, filterLength)
        #self.model['bias2'] = self.bias_scale * np.random.randn(numFilters)
        #self.model['weights3'] = self.weight_scale * np.random.randn(numFilters, numFilters, 1, filterLength)
        #self.model['bias3'] = self.bias_scale * np.random.randn(numFilters)
        self.wordWeights = {}
        self.wordGrads = {}
        
        self.conv_param = {'stride': 1, 'pad': 0}
        self.pool_param = {'pool_height': 1, 'pool_width': 2, 'stride_h': 1, 'stride_w': 2}

    def SGD(self, learningRate=0.1, decayFactor=0.95, numIters=5, minFreq = 10):
        #First, put all training words in the dictionary, if they aren't there
        occurrences = collections.Counter()
        for review in self.Data.trainData:
            text = review['text']
            for word in text.split():
                occurrences.update({word : 1})
                if(not word in self.wordWeights):
                    self.wordWeights[word] = np.random.randn(1)*self.weight_scale
        for key in occurrences:
            if occurrences[key] < minFreq:
                del self.wordWeights[key]

        #print self.wordWeights

        lr = learningRate

        for iter in range(0, numIters):
            print "Iteration: %d" %(iter)
            sumLoss = 0
            numValid = 0
            for review in self.Data.trainData:
                star = review['stars']
                text = review['text']
                if len(text.split()) > 15:
                    #print star, text
                    loss = self.predict(text, star) #Calculate gradients on example
                    for key in self.model:
                        #Update weights
                        self.model[key] -= self.mult[key]*lr*self.grads[key]+ self.reg[key]*self.model[key]
                        #print self.grads[key]
                    for i in xrange(len(self.labelVec)):
                        #Update word weights
                        self.wordWeights[self.labelVec[i]] -= self.mult['words']*lr*self.dx[0,0,0,i] + self.reg['words']*self.wordWeights[self.labelVec[i]] 
                    sumLoss += loss
                    numValid += 1.0
            lr *= decayFactor

            print ("Mean loss: %.4f, Learning Rate: %.4f" %(sumLoss/numValid, lr)) 
        print self.model
        #print self.dx
        #print self.wordWeights

    def test(self, verbose = False):
        predictions = []
        truths = []
        for review in self.Data.trainData:
            if(len(review['text'].split()) > 15):
                predictions.append(int(round(self.predict(review['text']))))
            else:
                predictions.append(4)
            truths.append(review['stars'])
        wrong = 0
        total = 0
        err = 0
        for i in xrange(len(predictions)):
            err += (predictions[i] - truths[i])**2
            if(predictions[i] != truths[i]):
                wrong += 1
            total += 1
        print "Train Misclassification Rate: ", float(wrong)/float(total) 
        print "Train RMSE: ", (float(err)/float(total))**(0.5)


        predictions = []
        truths = []
        for review in self.Data.testData:
            if(len(review['text'].split()) > 15):
                predictions.append(int(round(self.predict(review['text']))))
            else:
                predictions.append(4)
            truths.append(review['stars'])
        wrong = 0
        total = 0
        err = 0
        for i in xrange(len(predictions)):
            err += (predictions[i] - truths[i])**2
            if(predictions[i] != truths[i]):
                wrong += 1
            total += 1
        print "Test Misclassification Rate: ", float(wrong)/float(total) 
        print "Test RMSE: ", (float(err)/float(total))**(0.5)


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
        self.labelVec = []
        for word in text.split():
            if(word in self.wordWeights):
                #valVec.append(self.nGramWeights[word])
                valVec.append(self.wordWeights[word])
                self.labelVec.append(word)

        
        #print valVec

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

        n = len(text.split())
        a = 0.001*np.random.randn(1, 1, 1, n)
        #print n
        #print np.sum(a)
        gradval = 0.00001

        origLoss= self.predict2(a, star) #Calculate gradients on example
        dx = self.dx
        self.checkdx = dx*0
        originalGrads = self.grads
        for (w,x,y,z), val in np.ndenumerate(a):
            a[w,x,y,z] += gradval
            loss = self.predict2(a, star)
            self.checkdx[w,x,y,z] = (loss - origLoss)/gradval
            a[w,x,y,z] -= gradval
        error = np.mean(np.abs(dx - self.checkdx))/np.mean(np.abs(dx))
        print dx, self.checkdx
        print "dx Mean Error: ", error

        for key in self.model:
            self.checkGrads[key] = 0*self.model[key]
            error = 0
            
            if(len(self.model[key].shape) == 4):
                #print "here"
                for (w,x,y,z), val in np.ndenumerate(self.model[key]):
                    #print "step"
                    #print self.model[key][w,x,y,z]
                    self.model[key][w,x,y,z] += gradval
                    #print self.model[key][w,x,y,z]
                    loss = self.predict2(a, star)
                    #print loss
                    #print origLoss
                    #print (loss - origLoss)/gradval
                    self.checkGrads[key][w,x,y,z] = (loss - origLoss)/gradval
                    self.model[key][w,x,y,z] -= gradval
            elif(len(self.model[key].shape) == 1):
                for (w), val in np.ndenumerate(self.model[key]):
                    self.model[key][w] += gradval
                    loss = self.predict2(a,star)
                    self.checkGrads[key][w] = (loss - origLoss)/gradval
                    self.model[key][w] -= gradval
            error = np.mean(np.abs(originalGrads[key] - self.checkGrads[key]))/np.mean(np.abs(originalGrads[key]))
            #print "Original grads", originalGrads[key]
            #print "Numerical Grads", self.checkGrads[key]
            print key, "Mean Error: ", error
        #print originalGrads
        #print self.checkGrads


    def Pass(self, valMat, truth):

        W1, b1 = self.model['weights1'], self.model['bias1']
        #W2, b2 = self.model['weights2'], self.model['bias2']
        #W3, b3 = self.model['weights3'], self.model['bias3']
        N, C, H, W = valMat.shape

        #Forward Pass
        x = valMat
        x, cache1 = self.Conv_Forward(x, W1, b1)
        #x, cache2 = self.Max_Pool_Forward(x)
        #x, cache3 = self.Conv_Forward(x, W2, b2)
        #x, cache4 = self.Max_Pool_Forward(x)
        #x, cache5 = self.Conv_Forward(x, W3, b3)
        #x, cache6 = self.Max_Pool_Forward(x)


        if(truth == None):
            return self.Mean(x)

        #Backward Pass
        loss, dout = self.Mean_Loss(x, truth)
        #print "after Loss ", c1.shape
        #print c1

        #dout = self.Abs_Max_Pool_Backward(dout, cache6)
        #dout,  dW3, db3 = self.Conv_Backward(dout, cache5)
        #dout = self.Abs_Max_Pool_Backward(dout, cache4)
        #dout,  dW2, db2 = self.Conv_Backward(dout, cache3)
        #dout = self.Abs_Max_Pool_Backward(dout, cache2)
        dout,  dW1, db1 = self.Conv_Backward(dout, cache1)


        self.dx = dout
        self.grads.update({'weights1': dW1, 'bias1': db1})
        #self.grads.update({'weights2': dW2, 'bias2': db2})
        #self.grads.update({'weights3': dW3, 'bias3': db3})

        return self.Mean_Loss(x, truth)[0]

    def Pass2(self, valMat, truth):

        #W1, b1, W2, b2 = self.model['weights1'], self.model['bias1'], self.model['weights2'], self.model['bias2']
        W1, b1 = self.model['weights1'], self.model['bias1']
        W2, b2 = self.model['weights2'], self.model['bias2']
        W3, b3 = self.model['weights3'], self.model['bias3']
        N, C, H, W = valMat.shape

        #Forward Pass
        #Set1 
        #print "Forward Shapes"
        #print "Input ", valMat.shape, valMat
        #a1, cache1 = self.Conv_Forward(valMat, W1, b1)
        #print self.model['weights1']
        #print "conv ", a1.shape, a1
        a1 = valMat
        #a2, cache2 = self.ReLu_Forward(a1)
        #print "ReLu ", a2.shape, a2
        a2, cache3 = self.Max_Pool_Forward(a1)
        #print a3
        #print "pool ", a3.shape, a3


        if(truth == None):
            return self.Mean(a2)

        #Backward Pass
        #print "Backward Shapes"
        loss, c1 = self.Mean_Loss(a2, truth)
        #print "after Loss ", c1.shape
        #print c1

        #c5 = self.Max_Pool_Backward(c1, cache6)
        #print "after depool ", c2.shape
        #print c2
        #c6 = self.ReLu_Backward(c5, cache5)
        #print "after deRelu ", c3.shape
        #print c3
        #c7,  dW2, db2 = self.Conv_Backward2(c6, cache4)

        c2 = self.Max_Pool_Backward(c1, cache3)
        #print "after depool ", c2.shape
        #print c2
        #c2 = self.ReLu_Backward(c1, cache2)
        #print "after deRelu ", c3.shape
        #print c3
        #c4,  dW1, db1 = self.Conv_Backward(c1, cache1)
        #print "after deconv ", c4.shape
        #print c4

        self.dx = c2
        #self.grads = {'weights1': dW1, 'bias1': db1}

        return self.Mean_Loss(a2, truth)[0]


    def backprop(self, text):
            score, Inter1, inter2, inter3, Inter4 = forwardPass(text)

    def predict(self, text, truth=None):
        return self.Pass(self.getWordValueVector(text), truth)

    def predict2(self, a, truth=None):
        return self.Pass2(a, truth)

    def ReLu_Forward(self, x):
        #out = np.maximum(0, x)
        #print x
        out = np.where(x > 0, x, 0.001*x)
        cache = x
        #print out
        return out, cache

    def ReLu_Backward(self, dout, cache):
        x = cache
        dx = np.where(x > 0, dout, 0.001*dout)
        return dx

    def Mean_Loss(self, x, y):
        loss = 0.5*(self.Mean(x) - y)**2
        dx = np.zeros(x.shape) + (self.Mean(x) - y)/(np.prod(x.shape))
        #dx = x*0.0 + (self.Mean(x) - y)/(np.sum(np.where(x > 0))+ 0.1)
        #return loss, (dx*0 + 1)
        return loss, dx

    def Mean(self, x):
        return np.mean(x)
        #return np.sum(x)/(np.sum(np.where(x > 0))+ 0.1)

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
                for f in xrange(w.shape[0]): #For every filter
                    filt = w[f,j,:,:]
                    out[i,f,:,:] += signal.correlate2d(temp, filt, mode='same', boundary='fill', fillvalue=0) + b[f] #/x.shape[1]
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

        dw = 0.0*w
        dx = 0.0*x
        wmask = np.ones([H - HH + 1, W - WW + 1])


        for i in xrange(x.shape[0]): #For every data sample
            for j in xrange(x.shape[1]): #For every color
                dtemp = dout[i,j,:,:] #store layer to make things simpler
                dtemp2 = np.pad(dtemp, ((0,0),((WW-1)/2, (WW-1)/2)), 'constant',constant_values=0)
                xtemp = x[i,j,:,:]
                for f in xrange(w.shape[0]): #For every filter
                    filt = w[f,j,:,:]
                    dw[f,j,:,:] += signal.convolve2d(dtemp2, xtemp, mode='valid')
                    dx[i,j,:,:] += signal.convolve2d(dtemp, filt, mode='same', boundary='fill', fillvalue=0)

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
        mask = 0.0*x

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
                            dx[i,j,k,l] = dout[i,j,m,n]
                            n += 1
                            if(n == W):
                                n = 0
                                m += 1
        return dx

    def Abs_Max_Pool_Forward(self, x):
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
        mask = 0.0*x

        for i in xrange(N): #For each datapoint
            for j in xrange(C): #For each color
                k = 0
                while k < (out.shape[2]):
                    l = 0
                    while l < (out.shape[3]):
                        temp = x[i,j,k*sth:k*sth + ph,l*stw:l*stw + pw]
                        a = np.argmax(np.absolute(temp))
                        b = np.unravel_index(a, temp.shape)
                        out[i,j,k,l] = temp[b]
                        mask[i,j,b[0]+k*sth,b[1]+l*stw] = 1
                        l += 1
                    k += 1
        cache = mask
        return out, cache

    def Abs_Max_Pool_Backward(self, dout, cache):
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
                            dx[i,j,k,l] = dout[i,j,m,n]
                            n += 1
                            if(n == W):
                                n = 0
                                m += 1
        return dx