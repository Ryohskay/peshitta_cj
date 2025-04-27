"""Definition of the match between the algorithms and their name."""

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator

from src.classifier.wrappers import BoWEstimator

ALGORITHMS = {
    "mnb": {"algo": MultinomialNB, "description": "Multinomial Naive Bayes"},
    "knn": {"algo": KNeighborsClassifier, "description": "K-Nearest Neighbors"},
    "rf": {"algo": RandomForestClassifier, "description": "Random Forest"},
}

def get_algo_by_name(name: str, **kwargs) -> BaseEstimator: # noqa: ANN003
    """Get the algorithm by its name.

    Args:
        name: The name of the algorithm.
        kwargs: Additional arguments to pass to the initialiser of
            the algorithm.

    Returns:
        An object for the algorithm, an instance of `BaseEstimator` from
        scikit-learn.

    Raises:
        ValueError: If the algorithm is not defined.
    """
    if name not in ALGORITHMS:
        msg = (f"Algorithm {name} is not defined. Available algorithms are: "
        + f"{[(algo, ALGORITHMS[algo]["description"]) for algo in ALGORITHMS]}")
        raise ValueError(msg)
    return ALGORITHMS[name]["algo"](**kwargs)
