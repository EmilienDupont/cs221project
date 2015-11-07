def readCommonWords(inputFile):
    """
    Reads in the file pointed to by |inputFile| and returns a set of the most common words in
    the English language
    """
    commonWords = set()
    f = open(inputFile)
    for line in f:
        word = line.strip()
        commonWords.add(word)
    f.close()
    
    return commonWords

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

def wordFeaturesNoCommonWords(commonWords):
    """
    Returns a word feature extractor that doesn't take into account the most common words
    in the English language.
    Also, unlike normal extractor, this one doesn't distinguish between uppercase and lowercase
    words unless a word is in all caps
    """
    def extractor(text):
        wordCount = {}
        for word in text.split():
            upperCount = sum(1 if c.isupper() else 0 for c in word)
            lowerWord = word.lower()
            if lowerWord in commonWords:
                continue
            elif upperCount <= 1:
                if lowerWord in wordCount:
                    wordCount[lowerWord] += 1
                else:
                    wordCount[lowerWord] = 1
            else:
                if word in wordCount:
                    wordCount[word] += 1
                else:
                    wordCount[word] = 1
        return wordCount
    return extractor

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

def stemmedWordFeatures(leafWords):
    """
    Returns a function that return stemmed word features according to leafWords.
    E.g. if leafWords = ['s','es','ed','er','ly','ing'] then stem('orderly') = 'order',
    stem('catching') = 'catch'. So for example
    "Well priced prices" -> {'Well' : 1, 'pric' : 1}
    """
    def stem(word):
        for leaf in leafWords:
            if word[-len(leaf):] == leaf:
                return word[:-len(leaf)]
        return word

    def stemmedWordCount(text):
        wordCount = {}
        for word in text.split():
            stemWord = stem(word)
            if stemWord in wordCount:
                wordCount[stemWord] += 1
            else:
                wordCount[stemWord] = 1
        return wordCount

    return stemmedWordCount

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
