import json
import os.path

class Training:
    """
    Training class to load and train data.
    """
    def __init__(self, filename):
        """
        Loads in json data from file at filename.
        """
        if not os.path.isfile(filename):
            raise RuntimeError, "The file '%s' does not exist" % filename

        self.data = json.loads(open(filename).read())

    def letMeSeeThatData(self):
        print self.data

    def averageRating(self):
        """
        Calculates average star rating (from 1 to 5) of reviews in dataset.
        """
        return self.data['stars'] # Only one review right now

#train = Training('yelp_mini_set.json')
#train.letMeSeeThatData()
#print train.averageRating()
