import os.path
import json

def loadData(filename, numLinesData, numLinesTest):

    if not os.path.isfile(filename):
            raise RuntimeError, "The file '%s' does not exist" % filename

    trainData = []
    testData = []
    lineNum = 0

    with open(filename) as f:
        for line in f:
            if lineNum < numLinesData:
                trainData.append(json.loads(line))
                lineNum += 1
            elif lineNum < numLinesData + numLinesTest:
                testData.append(json.loads(line))
                lineNum += 1
            else:
                break

    return (trainData, testData)

def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale

def evaluatePredictor(examples, predictor):
    '''
    predictor: a function that takes an x and returns a predicted y.
    Given a list of examples (x, y), makes predictions based on |predict| and returns the fraction
    of misclassiied examples.
    '''
    error = 0
    for x, y in examples:
        if predictor(x) != y:
            error += 1
    return 1.0 * error / len(examples)
