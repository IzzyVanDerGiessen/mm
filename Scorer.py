import numpy as np
from scipy.spatial.distance import cosine


def singleFeatureScorer(test_feature, query_feature):
    #print(test_feature.shape, query_feature.shape)
    test_feature = test_feature.flatten()
    query_feature = query_feature.flatten()
    lengthLimiter = min(len(test_feature), len(query_feature)) #since sometimes we end up with weird numbers of frames (cuz of end of vid?)
    score = np.linalg.norm(test_feature[:lengthLimiter] - query_feature[:lengthLimiter])
    #similarity = 1 - cosine(test_feature, query_feature)
    return score #lower score is better








def feature_scorer(test_features, query_features):
    if len(test_features) != len(query_features):
        raise Exception("Not matching sizes of test and query features")
    scores = []
    for x, y in zip(test_features, query_features):
        scores.append(singleFeatureScorer(x,y))
    return np.mean(scores)
