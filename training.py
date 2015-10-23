import json
import os.path

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

train = Training('../../../../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json',10000)
#train.letMeSeeThatData()
print 'Average rating of %s reviews is %s' % (train.numLines, train.averageRating())
