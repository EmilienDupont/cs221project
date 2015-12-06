import numpy as np
from sklearn.cluster import KMeans

class ClusterWords():
    def __init__(self, embeddings, dictionary, numClusters=100):
        self.kmeans = KMeans(numClusters)
        self.numClusters = numClusters
        self.embeddings = embeddings
        self.reverseDictionary = { value: key for key, value in dictionary.items() }
        self.createClusters()
        self.wordToCluster = {}

    def createClusters(self):
        """
        final_embeddings should be an array of words as vectors (using skip gram model)
        rows: word
        columns: "features"
        This creates |numClusters| clusters using kmeans on the |embeddings| training set
        """
        print "Creating clusters using Kmeans..."
        self.kmeans.fit(self.embeddings)
        print "Finished creating clusters!"

    def getClusters(self, featureVectors):
        """
        featureVectors is an array where each row corresponds to a vector which has the same dimension
        as specified by |embeddings|.
        Returns a list of labels for each word.
        E.g. "Hello world" would be transformed into [[0.3, 0.4], [1,-0.2]], which would return
        e.g. [2,4] as the labels (one for each word)
        """
        return self.kmeans.predict(featureVectors)

    def getWordToCluster(self):
        """
        Returns a dictionary as (key: word, value: cluster to which it belongs)
        """
        self.wordToCluster = {}
        clusterLists = self.getClusters(self.embeddings)
        for i, cluster in enumerate(clusterLists):
            word = self.reverseDictionary[i]
            self.wordToCluster[word] = cluster
        return self.wordToCluster

    def printWordsInCluster(self, clusterNum):
        """
        Prints all word in cluster |clusterNum|
        """
        if clusterNum >= self.numClusters:
            print "There are only %s clusters" % self.numClusters
            return
        print "Words in cluster", clusterNum
        for word in wordToCluster:
            if wordToCluster[word] == 5:
                print word
        return
