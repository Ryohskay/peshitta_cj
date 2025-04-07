import numpy as np
from sklearn.naive_bayes import MultinomialNB

from classifier.wrappers import BoWEstimator

if __name__ == "__main__":
    mnb = BoWEstimator(MultinomialNB(), formatter=" ".join)
    y = np.append(np.ones(20), np.zeros(80))
    mnb.cross_validate(np.arange(100), y)
