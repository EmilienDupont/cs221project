import pickle
import random
import re
import string
import sys
import unicodedata
import createWordClusters

# table to store unicode punctuation characters
tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
                      if unicodedata.category(unichr(i)).startswith('P'))

# intermediate things to be used later in clause-level negation
negation_patterns = "(?:^(?:never|no|nothing|nowhere|noone|none|not|havent|hasn\'t|hadn\'t|can\'t|couldn\'t|shouldn\'t|" + \
                    "won\'t|wouldn\'t|don\'t|doesn\'t|didn\'t|isn\'t|aren\'t|ain\'t)$)|n't"

def neg_match(s):
    return re.match(negation_patterns, s, flags=re.U)

punct_patterns = "[.:;,!?]"

def punct_mark(s):
    return re.search(punct_patterns, s, flags=re.U)


def posNegClusterFeatures(embeddingsFile, dictionaryFile, lexiconFile, numClusters=100):
    """
    Returns a feature extractor which clusters words and then classifies them into
    positive and negative clusters. E.g. {Cluster1_NEG: 3, Cluster17_POS: 1} etc...
    """
    # Load the pickle files
    embeddings = pickle.load( open( embeddingsFile, "rb" ) )
    dictionary = pickle.load( open( dictionaryFile, "rb" ) )
    # Create a cluster object and return a dictionary mapping words to clusters
    cluster = createWordClusters.ClusterWords(embeddings, dictionary, numClusters)
    wordToCluster = cluster.getWordToCluster()

    # Read in the lexicon
    lexicon = readLexicon(lexiconFile)

    def extractor(text):
        featureVector = {}
        for word in text.split():
            if word in wordToCluster and word in lexicon:
                cluster = wordToCluster[word]
                polarity = int(lexicon[word]) # if 1 => positive, if 0 => negative
                if polarity:
                    featureName = "Cluster" + str(cluster) + "_POS"
                else:
                    featureName = "Cluster" + str(cluster) + "_NEG"

                if featureName in featureVector:
                    featureVector[featureName] += 1
                else:
                    featureVector[featureName] = 1
        return featureVector

    return extractor

def clauseClusterFeatures(embeddingsFile, dictionaryFile, lexiconFile, numClusters=100):
    """
    Returns a feature extractor which clusters words and then classifies them into
    positive and negative clusters. E.g. {Cluster1_NEG: 3, Cluster17_POS: 1} , and also takes
    into account clause-level negation
    """
    # Load the pickle files
    embeddings = pickle.load( open( embeddingsFile, "rb" ) )
    dictionary = pickle.load( open( dictionaryFile, "rb" ) )
    # Create a cluster object and return a dictionary mapping words to clusters
    cluster = createWordClusters.ClusterWords(embeddings, dictionary, numClusters)
    wordToCluster = cluster.getWordToCluster()

    # Read in the lexicon
    lexicon = readLexicon(lexiconFile)

    def extractor(text):
        featureVector = {}
        prevNeg = False
        for word in text.split():
            if word in wordToCluster and word in lexicon:
                cluster = wordToCluster[word]
                polarity = int(lexicon[word]) # if 1 => positive, if 0 => negative
                if not prevNeg:
                    prevNeg = (neg_match(word) != None)
                    if polarity:
                        featureName = "Cluster" + str(cluster) + "_POS"
                    else:
                        featureName = "Cluster" + str(cluster) + "_NEG"
                    if featureName in featureVector:
                        featureVector[featureName] += 1
                    else:
                        featureVector[featureName] = 1
                else:
                    if polarity:
                        featureName = "Cluster" + str(cluster) + "_NEG"
                    else:
                        featureName = "Cluster" + str(cluster) + "_POS"
                    if featureName in featureVector:
                        featureVector[featureName] += 1
                    else:
                        featureVector[featureName] = 1
            if punct_mark(word):
                prevNeg = False
        return featureVector

    return extractor    

def clusterFeatures(embeddingsFile, dictionaryFile, numClusters=100):
    """
    Returns a feature extractor which using the skip gram embeddings clusters
    various words using k-means returns the review as a dictionary where
    key: cluster, value: number of words belonging to that cluster
    """
    # Load the pickle files
    embeddings = pickle.load( open( embeddingsFile, "rb" ) )
    dictionary = pickle.load( open( dictionaryFile, "rb" ) )
    # Create a cluster object and return a dictionary mapping words to clusters
    cluster = createWordClusters.ClusterWords(embeddings, dictionary, numClusters)
    wordToCluster = cluster.getWordToCluster()

    def extractor(text):
        featureVector = {}
        for word in text.split():
            if word in wordToCluster:
                cluster = wordToCluster[word]
                if cluster in featureVector:
                    featureVector[cluster] += 1
                else:
                    featureVector[cluster] = 1
        return featureVector

    return extractor


def readLexicon(inputFile):
    """
    Reads in the lexicon file pointed to by |inputFile| and returns a dictionary
    of words: if the word is positive its value is 1, if it is negative its value
    is 0. For example:
    {Good: 1, Excellent: 1, Terrible: 0}
    This is based on the NRC Emotion Lexicon v0.92
    """
    f = open(inputFile)
    lexicon = {}
    lineCount = 0

    print "Creating Lexicon..."

    # Note that format of file is:
    # word, emotion, true/false
    # There are 10 emotions for each word

    for line in f:
        lineCount += 1
        # 6th line corresponds to negative
        if lineCount % 10 == 6:
            word, emotion, boolean = line.split()
            if int(boolean):
                lexicon[word] = 0
        # 7th line corresponds to positive
        elif lineCount % 10 == 7:
            word, emotion, boolean = line.split()
            if int(boolean):
                lexicon[word] = 1
    f.close()

    print "Created lexicon!"

    return lexicon

def readFullLexicon(inputFile):
    """
    Reads in the lexicon file pointed to by |inputFile| and returns a dictionary
    of words: the key is the word and the value is a list of numbers corresponding to
    various emotions.
    0 anger
    1 anticipation
    2 disgust
    3 fear
    4 joy
    5 negative
    6 positive
    7 sadness
    8 surprise
    9 trust
    {Good: [4,6], Excellent: [6], Terrible: [2,5]}
    This is based on the NRC Emotion Lexicon v0.92
    """
    f = open(inputFile)
    lexicon = {}
    lineCount = 0

    print "Creating Full Lexicon..."

    # Note that format of file is:
    # word, emotion, true/false
    # There are 10 emotions for each word

    for line in f:
        word, emotion, boolean = line.split()
        if int(boolean):
            if word not in lexicon:
                lexicon[word] = [lineCount % 10]
            else:
                lexicon[word].append(lineCount % 10)
        lineCount += 1
    f.close()

    print "Created full lexicon!"

    return lexicon


def positiveNegativeCounts(inputFile):
    """
    Returns funciton that returns a two dimensional feature vector:
    -Count of positive words (based on lexicon)
    -Count of negative words (based on lexicon)
    """
    lexicon = readLexicon(inputFile)
    def extractor(text):
        featureVector = {'-POSITIVE-': 0, '-NEGATIVE-': 0}
        for word in text.split():
            if word in lexicon:
                if lexicon[word]:
                    featureVector['-POSITIVE-'] += 1
                else:
                    featureVector['-NEGATIVE-'] += 1
        return featureVector

    return extractor

def emotionCounts(inputFile):
    """
    Returns funciton that returns a 10 dimensional feature vector which
    contains the counts of various emotions
    """
    numberToToken = {0: '-ANGER-', 1: '-ANTICIPATION-', 2: '-DISGUST-', 3: '-FEAR-',
                     4: '-JOY-', 5: '-NEGATIVE-', 6: '-POSITIVE-', 7: '-SADNESS-',
                     8: '-SURPRISE-', 9 : '-TRUST-'}

    lexicon = readFullLexicon(inputFile)

    def extractor(text):
        featureVector = {'-ANGER-': 0, '-ANTICIPATION-': 0, '-DISGUST-': 0,
                         '-FEAR-': 0, '-JOY-': 0, '-NEGATIVE-': 0, '-POSITIVE-': 0,
                         '-SADNESS-': 0, '-SURPRISE-': 0, '-TRUST-': 0}
        for word in text.split():
            if word in lexicon:
                for number in lexicon[word]:
                    featureVector[numberToToken[number]] += 1
        return featureVector

    return extractor


def removePunctuation(text):
    """
    Function that removes the punctuation from a unicode formatted input string
    """
    return text.translate(tbl).lower()1

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
        # strip a string of punctuation marks
        wordStripped = removePunctuation(word)
        if wordStripped in wordCount:
            wordCount[wordStripped] += 1
        else:
            wordCount[wordStripped] = 1
    return wordCount

def notManyFeatures(text):
    """
    Function to return the word count in a string as a dict (with some probability)
    """
    wordCount = {}
    for word in text.split():
        if random.random() > .9:
            # strip a string of punctuation marks
            wordStripped = removePunctuation(word)
            if wordStripped in wordCount:
                wordCount[wordStripped] += 1
            else:
                wordCount[wordStripped] = 1
    return wordCount

smileyList = [':)', ':(', ':D', ':P', ":'(", '>: (']

def wordFeaturesWithSmileys(text):
    """
    Takes into account smileys
    """
    wordCount = {}
    for word in text.split():
        for smiley in smileyList:
            if word.find(smiley) >= 0:
                if smiley in wordCount:
                    wordCount[smiley] += 1
                else:
                    wordCount[smiley] = 1
        # strip a string of punctuation marks
        wordStripped = removePunctuation(word)
        if wordStripped in wordCount:
            wordCount[wordStripped] += 1
        else:
            wordCount[wordStripped] = 1
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
            wordStripped = removePunctuation(word)
            upperCount = sum(1 if c.isupper() else 0 for c in wordStripped)
            lowerWord = wordStripped.lower()
            if lowerWord in commonWords:
                continue
            elif upperCount <= 1:
                if lowerWord in wordCount:
                    wordCount[lowerWord] += 1
                else:
                    wordCount[lowerWord] = 1
            else:
                if wordStripped in wordCount:
                    wordCount[wordStripped] += 1
                else:
                    wordCount[wordStripped] = 1
        return wordCount
    return extractor

def wordFeaturesWithNegation(commonWords, leafWords):
    """
    Word feature extractor that tries to take into account the effects of negation and clause-level punctuation
    marks and tokenizes the appropriate negated words. Does not use them if they happen to be common words, and
    considers only stemmed words
    """
    def stem(word):
        for leaf in leafWords:
            if word[-len(leaf):] == leaf:
                return word[:-len(leaf)]
            else:
                return word

    def extractor(text):
        wordCount = {}
        prevNeg = False
        for word in text.split():
            wordStripped = stem(removePunctuation(word))
            if not prevNeg:
                prevNeg = (neg_match(word) != None)
                if not wordStripped in commonWords:
                    if wordStripped in wordCount:
                        wordCount[wordStripped] += 1
                    else:
                        wordCount[wordStripped] = 1
            else:
                if not wordStripped in commonWords:
                    negWord = wordStripped + "_NEG"
                    if negWord in wordCount:
                        wordCount[negWord] += 1
                    else:
                        wordCount[negWord] = 1
            if punct_mark(word):
                prevNeg = False
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
            stemWord = stem(removePunctuation(word))
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
