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
    def __init__(self, filterLength = 3, numFilters = 4):
        self.filterLength = filterLength
        self.numFilters = numFilters
        self.Data = []
        self.nGramWeights = []
        #self.weights1 = np.zeros((numFilters,filterLength)) #filterLength x numFilters
        #self.weights2 = np.zeros((numFilters,filterLength)) #filterLength x numFilters
        weight_scale = 0.001
        bias_scale = 0.00001;

        self.grads = {}
        self.model = {}
        self.model['weights1'] = weight_scale * np.random.randn(numFilters, 1, 1, filterLength)
        self.model['bias1'] = bias_scale * np.random.randn(numFilters)
        self.model['weights2'] = weight_scale * np.random.randn(numFilters, numFilters, 1, filterLength)
        self.model['bias2'] = bias_scale * np.random.randn(numFilters)
        
        self.conv_param = {'stride': 1, 'pad': (self.filterLength - 1) / 2}
        self.pool_param = {'pool_height': 1, 'pool_width': 2, 'stride_h': 1, 'stride_w': 2}

    def SGD(self, learningRate=0.001, decayFactor=0.99, numIters=1):
        lr = learningRate

        for iter in range(0, numIters):
            for review in self.Data.trainData:
                star = review['stars']
                text = review['text']
                if len(text.split()) > 5:
                    #print star, text
                    self.predict(text, star) #Calculate gradients on example
                    for key in self.model:
                        self.model[key] -= lr*self.grads[key]
                    lr *= decayFactor
                        #print key

        print self.model
        print "done"

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
        if(len(valVec)%2 == 0):
            valMat = np.zeros([1,1,1,len(valVec)])
            for i in range(0, len(valVec)):
                valMat[0,0,0,i] = valVec[i]
        else:
            valMat = np.zeros([1,1,1,len(valVec)-1])
            for i in range(0, len(valVec)-1):
                valMat[0,0,0,i] = valVec[i]

        return valMat

    def loadWeights(self, unigramModel):
        self.nGramWeights = unigramModel.weights
        self.Data = unigramModel.Data

    def Pass(self, valMat, truth):

        W1, b1, W2, b2 = self.model['weights1'], self.model['bias1'], self.model['weights2'], self.model['bias2']
        N, C, H, W = valMat.shape

        #Forward Pass
        #Set1 
        #print "Forward Shapes"
        #print "Input ", valMat.shape
        a1, cache1 = self.Conv_Forward(valMat, W1, b1)
        #print "conv ", a1.shape
        a2, cache2 = self.ReLu_Forward(a1)
        #print "ReLu ", a2.shape
        a3, cache3 = self.Max_Pool_Forward(a2)
        #print "pool ", a3.shape
        #Set 2   
        a4, cache4 = self.Conv_Forward(a3, W2, b2)
        #print "conv ",a4.shape
        a5, cache5 = self.ReLu_Forward(a4)
        #print "ReLu ", a5.shape
        a6, cache6 = self.Max_Pool_Forward(a5)
        #print "Pool ", a6.shape
        if(truth == None):
            return self.Mean(a6)

        #Backward Pass
        #print "Backward Shapes"
        loss, c1 = self.Mean_Loss(a6, truth)
        #print "after Loss ", c1.shape
        c2 = self.Max_Pool_Backward(c1, cache6)
        #print "after depool ", c2.shape
        c3 = self.ReLu_Backward(c2, cache5)
        #print "after deRelu ", c3.shape
        c4,  dW2, db2 = self.Conv_Backward(c3, cache4)
        #print "after deconv ", c4.shape
        #Set 2
        c5 = self.Max_Pool_Backward(c4, cache3)
        #print "after depool ", c5.shape
        c6 = self.ReLu_Backward(c5, cache2)
        #print "after ReLu", c6.shape
        c7,  dW1, db1 = self.Conv_Backward(c6, cache1)
        #print "after deconv ", c7.shape

        self.grads = {'weights1': dW1, 'bias1': db1, 'weights2': dW2, 'bias2': db2}

        return self.Mean(a6), self.grads

    def backprop(self, text):
            score, Inter1, inter2, inter3, Inter4 = forwardPass(text)

    def predict(self, text, truth=None):
        return self.Pass(self.getWordValueVector(text), truth)

    def ReLu_Forward(self, x):
        out = np.maximum(0, x)
        cache = x
        return out, cache

    def ReLu_Backward(self, dout, cache):
        x = cache
        dx = np.where(x > 0, dout, 0)
        return dx

    def Mean_Loss(self, x, y):
        loss = (self.Mean(x) - y)**2
        dx = x*0.0 + loss/(np.prod(x.shape))
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
        W = x.shape[3] + 2*pad
        HH = w.shape[2]
        WW = w.shape[3]
        N = x.shape[0]
        F = w.shape[0]
        l = 0
        k = 0

        out = np.zeros((N,F,(H-HH)/stride+1, (W-WW)/stride+1))

        for i in xrange(x.shape[0]): #For every data sample
            for j in xrange(x.shape[1]): #For every color
                temp = x[i,j,:,:] #store layer to make things simpler
                temp = np.pad(temp, pad, 'constant')
                for f in xrange(w.shape[0]): #For every filter
                    filt = w[f,j,:,:]
                    k = 0
                    while k < (temp.shape[0]-2*pad):
                        l = 0
                        while l < (temp.shape[1]-2*pad):
                            seg = temp[k:k + HH,l:l + WW]
                            out[i,f,k/stride,l/stride] += np.sum(seg*filt) + b[f]/x.shape[1]
                            l += stride
                        k += stride
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
        W = x.shape[3] + 2*pad
        HH = w.shape[2]
        WW = w.shape[3]
        N = x.shape[0]
        F = w.shape[0]
        l = 0
        k = 0

        dw = 0*w
        dx = 0*x

        for i in xrange(x.shape[0]): #For every data sample
            for j in xrange(x.shape[1]): #For every color
                temp = x[i,j,:,:] #store layer to make things simpler
                temp = np.pad(temp, pad, 'constant')
                for f in xrange(w.shape[0]): #For every filter
                    k = 0
                    while k < (temp.shape[0]-2*pad): #For every position
                        l = 0
                        while l < (temp.shape[1]-2*pad): #For every position
                            for m in range(k,k+HH): #For every element
                                for n in range(l,l+WW): #For every element
                  #out[i,f,k/stride,l/stride] += np.sum(temp[k:k + HH,l:l + WW]*w[f,j,:,:]) + b[f]/x.shape[
                                    #print '-'
                                    #print dw.shape, dout.shape, w.shape 
                                    #print f, j, m-k, n-l
                                    #print i, f, k/stride,l/stride
                                    #print f, j, m-k, n-l 
                                    dw[f,j,m-k,n-l] += dout[i,f,k/stride,l/stride]*temp[m,n]
                                    if m <= x.shape[2] and n <= x.shape[3] and m >= pad and n >= pad:
                                        dx[i,j,m-pad,n-pad] += dout[i,f,k/stride,l/stride]*w[f,j,m-k,n-l]
                            l += stride
                        k += stride
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
        k = 0
        l = 0
        for i in xrange(N):
            for j in xrange(C):
                k = 0
                while k < (out.shape[2]):
                    l = 0
                    while l < (out.shape[3]):
                        out[i,j,k,l] = np.max(x[i,j,k*sth:k*sth + ph,l*stw:l*stw + pw])
                        l += 1
                    k += 1
        cache = x
        return out, cache

    def Max_Pool_Backward(self, dout, cache):
        """
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.

        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        [N,C,H,W ] = dout.shape
        x = cache
        ph = self.pool_param['pool_height']
        pw = self.pool_param['pool_width']
        sth = self.pool_param['stride_h']
        stw = self.pool_param['stride_w']
        out = np.zeros((N,C,H/ph, W/pw))
        dx = np.zeros((N,C,H*ph, W*ph))
        dx = np.zeros(x.shape)

        for i in xrange(N):
            for j in xrange(C):
                k = 0
                while k < (x.shape[2]):
                    l = 0
                    while l < (x.shape[3]-1):
                        mx = np.max(x[i,j,k:k+ph,l:l+pw])
                        for m in range(k,k+ph):
                            for n in range(l,l+pw):
                                if(x[i,j,m,n] >= mx):
                                    #print i, j, m, n
                                    #print i, j, np.floor(k/sth),np.floor(l/stw)
                                    #print l, stw, dout.shape, dx.shape
                                    dx[i,j,m,n] = dout[i,j,np.floor(k/sth),np.floor(l/stw)]
                                else:
                                    dx[i,j,m,n] = 0
                        l += stw
                    k += sth
        return dx