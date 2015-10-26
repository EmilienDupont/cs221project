def nGramFeatures(n):
    """
    Returns a function that returns "n-gram" features from a string
    """
    def nGramFunction(text):
        featureVec = {}
        string = text.replace(" ","")
        for i in range(len(string)-(n-1)):
            if string[i:i+n] in featureVec:
                featureVec[string[i:i+n]] += 1
            else:
                featureVec[string[i:i+n]] = 1
        return featureVec
    return nGramFunction

def wordFeatures(text):
    """
    Function to return the word count in a string as a dict.
    E.g. "This is the way it is" -> {'This' : 1, 'is' : 2, 'the' : 1, 'way' : 1, 'it' : 1}
    """
    wordCount = {}
    for word in text.split():
        if word in wordCount:
            wordCount[word] += 1
        else:
            wordCount[word] = 1
    return wordCount

def stemFunction(leafWords):
    """
    Returns a function that stems word which have an ending in leafwords.
    E.g. if leafWords = ['s','es','ed','er','ly','ing']
    Then stem('orderly') = 'order', stem('catching') = 'catch'
    """
    def stem(word):
        for leaf in leafWords:
            if word[-len(leaf):] == leaf:
                return word[:-len(leaf)]

    return stem
