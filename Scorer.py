import numpy as np


def singleFeatureScorer(test_feature, query_feature):
    lengthLimiter = min(len(test_feature), len(query_feature)) #since sometimes we end up with weird numbers of frames (cuz of end of vid?)
    score = np.abs(test_feature[:lengthLimiter] - query_feature[:lengthLimiter]).sum()
    return score

def multiModalFeatureScorer(test_features, query_features):
    if len(test_features) != len(query_features):
        raise Exception("Not matching sizes of test and query features")
    scores = []
    for x, y in zip(test_features, query_features):
        scores.append(singleFeatureScorer(x,y))
    return np.mean(scores)      

