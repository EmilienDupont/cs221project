import features
import random
import tensorflow as tf

from data import *

# Analyzing command line arguments
if len(sys.argv) < 2:
  print 'Usage:'
  print '  python %s <JSON file>' % sys.argv[0]
  exit()

inputFile = sys.argv[1]

# Import Data
reviews = Data(inputFile, numLines = 10000, testLines = 1000)
reviews.getInfo()
reviews.shuffle()

#lexicon = features.readFullLexicon('NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt')
featureExtractor = features.clusterFeatures("embeddings.p", "dictionary.p", 200)

# Create sparse numpy arrays
reviews.convertDataToArray(featureExtractor)
reviews.convertLabelsToOneHot() # Need this for tensorflow

#print reviews.trainArray.todense()
#print reviews.trainLabelOneHot.transpose().todense()

x = tf.placeholder("float", [None, reviews.numFeatures])

# We have 5 stars and reviews.numFeatures number of reviews
W = tf.Variable(tf.zeros([reviews.numFeatures, 5]))
b = tf.Variable(tf.zeros([5]))

# Our prediction
y = tf.nn.softmax(tf.matmul(x,W) + b)

# The correct answers
y_ = tf.placeholder("float", [None, 5])

# Define the cross entropy
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# What you are going to do when you train (0.01 is learning rate)
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

# Initialize all variables you created
init = tf.initialize_all_variables()

# Create a session
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

batchSize = 100

def getRandomIndices(numIndices, maxIndex):
    """
    Function to return a list of |numIndices| random indices, where the indices
    are at most |maxIndex|.
    """
    return [ random.randint(0, maxIndex - 1) for _ in range(numIndices) ]

# Train for a 1000 iterations
for i in range(5):
    # Nothing is actually changing, need to do mini batches!
  if True:
    print i
    print "test accuracy:"
    print sess.run(accuracy, feed_dict={x: reviews.testArray.todense(), y_: reviews.testLabelOneHot.transpose().todense()})
    print "train accuracy:"
    print sess.run(accuracy, feed_dict={x: reviews.trainArray.todense(), y_: reviews.trainLabelOneHot.transpose().todense()})
  #batch_xs, batch_ys = mnist.train.next_batch(100) # Add a batch of a 100 examples, with a hundred labels
  #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  randomIndices = getRandomIndices(batchSize, reviews.numLines)
  batch_xs = reviews.trainArray.todense()[randomIndices,:]
  batch_ys = reviews.trainLabelOneHot.todense().transpose()[randomIndices,:]
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  # What you feed in to the train_step is x (the training data and y_ (the true labels).
  # x allows you to calculate y, and y_ and y allow you to calculate cross_entropy
  # which in turn allows you to calculate the new train step
