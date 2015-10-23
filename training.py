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
        for i in range(numLines):
            self.data.append( json.loads(open(filename).readline()) )

    def letMeSeeThatData(self):
        print self.data

    def averageRating(self):
        """
        Calculates average star rating (from 1 to 5) of reviews in dataset.
        """
        return sum( review['stars'] for review in self.data )/self.numLines

train = Training('yelp_mini_set.json',1)
train.letMeSeeThatData()
print train.averageRating()
