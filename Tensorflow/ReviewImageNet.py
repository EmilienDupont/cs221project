import tensorflow.python.platform
import collections
import numpy as np
import tensorflow as tf
import re
import sys
import string
import pickle
import scipy.interpolate
import random
sys.path.insert(0, '../')
from data import *

# Analyzing command line arguments
if len(sys.argv) < 2:
  print ("Usage:")
  print ("  python %s <JSON file>" % sys.argv[0])
  exit()

inputFile = sys.argv[1]

reviews = Data(inputFile, numLines = 10000, testLines = 1000)
reviews.shuffle()

#Load embeddings of words
embeddings = pickle.load( open( "embeddings.p", "rb" ) )
dictionary = pickle.load( open( "dictionary.p", "rb" ) )

#Load all data into the proper format
def loadData(inputData):
  print "Loading dataset..."
  finalData = []
  finalGT = []
  for review in reviews.trainData:
    first = True
    for word in review['text'].split():
      wordT = word.encode("utf-8").lower()
      wordT = wordT.translate(None, string.punctuation)
      if(wordT in dictionary):
        num = dictionary[wordT]
        vec = np.expand_dims(np.array(embeddings[num]),1)
        #print vec.shape
        if(first):
          revVec = vec
        else:
          revVec = np.concatenate((revVec, vec), 1)
          #print "jdsfhls"
        first = False
    #Resample review to 64 length
    if(not first):
      ny, nx = revVec.shape
      x = np.linspace(0, 1, nx)
      y = np.linspace(0, 1, ny)
      xv, yv = np.meshgrid(x, y)
      xv = np.squeeze(np.reshape(xv,(1,-1)))
      yv = np.squeeze(np.reshape(yv,(1,-1)))
      revVec = np.squeeze(np.reshape(revVec,(1,-1)))
      ny, nx = (64, 64)
      x = np.linspace(0, 1, nx)
      y = np.linspace(0, 1, ny)
      xv2, yv2 = np.meshgrid(x, y)
      newRev = scipy.interpolate.griddata((xv,yv),revVec,(xv2,yv2),method='nearest')

      finalData.append(newRev)
      oneHot = [0,0,0,0,0]
      oneHot[review['stars']-1] = 1
      finalGT.append(oneHot)
  print "done-zo"
  return (finalData, finalGT)

#Load the training and test sets
allTrainingData = loadData(reviews.trainData)
allTestData = loadData(reviews.testData)

sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 64, 64])
y_ = tf.placeholder("float", shape=[None, 5])
print len(allTrainingData[0]), "Training"
print len(allTestData[0]), "Test"

sess.run(tf.initialize_all_variables())

#Function to return a batch from whatever dataset is being used (for training)
#Data is shuffled each time the data has been passed through, and 
def getBatch(dataSet, numItems, batchNum):
  Total = len(dataSet[0])
  #If we've gone through the dataset before, shuffle it
  if(batchNum*numItems % Total <= numItems):
    combined = zip(dataSet[0], dataSet[1])
    random.shuffle(combined)
    a, b = zip(*combined)
    dataSet = (a,b)
  #After that, get a batch ready
  batchData = []
  batchLabel = []

  startPoint = batchNum*numItems % Total
  for i in xrange(numItems):
    current = (startPoint + i) % Total
    batchData.append(dataSet[0][current])
    batchLabel.append(dataSet[1][current])
  return dataSet, (batchData, batchLabel)


#Functons from google to make layer definition simpler
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#Define out network, a standard double conv-relu-pool with a softmax
W_conv1 = weight_variable([64, 5, 1, 5])
b_conv1 = bias_variable([1])

x_image = tf.reshape(x, [-1,64,64,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([32, 5, 5, 5])
b_conv2 = bias_variable([1])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([16 * 16 * 5, 512])
b_fc1 = bias_variable([512])

h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*5])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([512, 5])
b_fc2 = bias_variable([5])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
rmse = tf.reduce_mean(tf.pow(tf.argmax(y_conv,1)-tf.argmax(y_,1),2))
sess.run(tf.initialize_all_variables())
for i in range(10000):
  allTrainingData, batch = getBatch(allTrainingData, 50, i)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#

right = 0 
total = 0

for i in xrange(len(allTrainData[0])):
  acc = accuracy.eval(feed_dict={x: allTestData[0][i:i+1], y_: allTestData[1][i:i+1], keep_prob: 1.0})
  right += acc
  total += 1
  i += 1
print "Train Accuracy: ", float(right)/float(total)

right = 0 
total = 0

for i in xrange(len(allTestData[0])):
  acc = accuracy.eval(feed_dict={x: allTestData[0][i:i+1], y_: allTestData[1][i:i+1], keep_prob: 1.0})
  right += acc
  total += 1
  i += 1
print "Test Accuracy: ", float(right)/float(total)
