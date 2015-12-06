import pickle
import numpy as np
from sklearn.cluster import KMeans

embeddings = pickle.load( open( "embeddings.p", "rb" ) )
dictionary = pickle.load( open( "dictionary.p", "rb" ) )

class ClusterWords():
    def __init__(self, embeddings, numClusters=100):
        self.kmeans = KMeans(numClusters)
        self.numClusters = numClusters
        self.createClusters(embeddings)

    def createClusters(self, embeddings):
        """
        final_embeddings should be an array of words as vectors (using skip gram model)
        rows: word
        columns: "features"
        This creates |numClusters| clusters using kmeans on the embeddings training set
        """
        print "Creating clusters using Kmeans..."
        self.kmeans.fit(embeddings)
        print "Finished creating clusters!"

    def getClusters(self, featureVectors):
        """
        featureVectors is an array where each row corresponds to a vector which has the same dimension
        as specified by embeddings.
        Returns a list of labels for each word.
        E.g. "Hello world" would be transformed into [[0.3, 0.4], [1,-0.2]], which would return
        e.g. [2,4] as the labels (one for each word)
        """
        return self.kmeans.predict(featureVectors)

cluster = ClusterWords(embeddings)
clusterLists = cluster.getClusters(embeddings)
wordToCluster = {}

#reverseDictionary = {value: key for key, value in dictionary.items()}

for i, cluster in enumerate(clusterLists):
    word = dictionary[i]
    wordToCluster[word] = cluster

# Print all words in some cluster
for word in wordToCluster:
    if wordToCluster[word] == 5:
        print word
