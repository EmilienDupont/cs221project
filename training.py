import json
import os.path
import collections
import sys

# Analysing command line arguments
if len(sys.argv) < 2:
  print 'Usage:'
  print '  python %s <JSON file>' % sys.argv[0]
  exit()

inputFile = sys.argv[1]

class Training:
    """
    Class to load and train data.
    """
    def __init__(self, filename, numLines=100):
        """
        Loads in json data from file at filename.
        """
        if not os.path.isfile(filename):
            raise RuntimeError, "The file '%s' does not exist" % filename

        self.numLines = numLines
        self.data = []

        lineNum = 0
        with open(filename) as f:
            for line in f:
                if lineNum < self.numLines:
                    self.data.append(json.loads(line))
                    lineNum += 1

    def letMeSeeThatData(self):
        print self.data

    def averageRating(self):
        """
        Calculates average star rating (from 1 to 5) of reviews in dataset.
        """
        return sum( float(review['stars']) for review in self.data )/self.numLines

    def modeRating(self):
        """
        Calculates most common rating of reviews
        """
        C = collections.Counter()
        for review in self.data:
            s = review['stars']
            C.update({s:1})
        return C.most_common(1)[0][0]

#train = Training('../../../../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json',10000)
train = Training(inputFile)
#train.letMeSeeThatData()
print 'Average rating of %s reviews is %s' % (train.numLines, train.averageRating())

print 'Mode rating of %s reviews is %s' % (train.numLines, train.modeRating())
