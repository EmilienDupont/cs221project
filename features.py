import re
import string
import sys
import unicodedata

# table to store unicode punctuation characters
tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
                      if unicodedata.category(unichr(i)).startswith('P'))

def removePunctuation(text):
    """
    Function that removes the punctuation from a unicode formatted input string
    """
    return text.translate(tbl)

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

negation_patterns = "(?:^(?:never|no|nothing|nowhere|noone|none|not|havent|hasn\'t|hadn\'t|can\'t|couldn\'t|shouldn\'t|" + \
                    "won\'t|wouldn\'t|don\'t|doesn\'t|didn\'t|isn\'t|aren\'t|ain\'t)$)|n't"

def neg_match(s):
    return re.match(negation_patterns, s, flags=re.U)

punct_patterns = "[.:;!?]"

def punct_mark(s):
    return re.search(punct_patterns, s, flags=re.U)

def wordFeaturesWithNegation(text):
    wordCount = {}
    prevNeg = False
    for word in text.split():
        if not prevNeg:
            prevNeg = (neg_match(word) != None)
            if word in wordCount:
                wordCount[word] += 1
            else:
                wordCount[word] = 1
        else:
            negWord = word + "_NEG"
            if negWord in wordCount:
                wordCount[negWord] += 1
            else:
                wordCount[negWord] = 1
        if punct_mark(word):
            prevNeg = False
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
