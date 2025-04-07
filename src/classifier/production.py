
from sklearn.naive_bayes import  MultinomialNB
from classifier.eval_utils import eval_and_save
from classifier.wrappers import BoWEstimator


if __name__ == "__main__":
    print("Plain MNB")
    BoWEstimator(MultinomialNB(), " ".join)
    BoWEstimator
