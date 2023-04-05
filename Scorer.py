import numpy as np


def singleFeatureScorer(test_feature, query_feature):
    test_feature = test_feature.flatten()
    query_feature = query_feature.flatten()
    lengthLimiter = min(len(test_feature), len(query_feature)) #since sometimes we end up with weird numbers of frames (cuz of end of vid?)
    test_feature = normalize_feature(test_feature[:lengthLimiter])
    query_feature = normalize_feature(query_feature[:lengthLimiter])
    score = np.mean(np.abs(test_feature - query_feature))
    return score

def normalize_feature(feature):
    return feature / np.linalg.norm(feature)


def feature_scorer(test_features, query_features):
    if len(test_features) != len(query_features):
        raise Exception("Not matching sizes of test and query features")
    scores = []
    for x, y in zip(test_features, query_features):
        scores.append(singleFeatureScorer(x,y))
    return np.mean(scores)
