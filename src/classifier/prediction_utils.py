"""Utility functions to make predictions with a classifier."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from classifier.result_utils import (
    Predictions,
    ProbaPredictions,
    Verse,
)
from classifier.wrappers import Classifier, ProbaClassifier


def predict(
        classifier: Classifier,
        test_samples: NDArray | list,
        test_labels: NDArray
    ) -> Predictions:
    """Predict on the data with a classifier and get some simple statistics.

    Args:
        classifier: any object that has a method `.predict()`.
        test_samples: inputs to the classifier
        test_labels: correct labels (gold references) for the inputs. This is
            not used to predict probabilities, but is used to calculate
            prediction accuracy etc.

    Returns:
        A :class:`Prediction` class instance.
    """
    y_pred = classifier.predict(test_samples)
    return Predictions(test_samples, y_pred,
                test_labels)


def convert(probas: NDArray | list[Any], thresh: float) -> NDArray:
    """Convert list of probabilities to a list of labels.

    Args:
        probas: list of predicted probabilities.
        thresh: threshold to decide if a probability prediction should be
            labelled as an instance of the class.

    Returns:
        0 if prediction for label 0 is over the threshold,
        1 if prediction for label 1 is over the threshold,
        -1 if probabilities for both labels do not exceed the threshold.

    Raises:
        ValueError: if neither of the classes score 0.5 probability.
    """
    result = np.empty(0, dtype=int)
    for i in range(len(probas)):
        probability = probas[i]
        if probability[0] > thresh:
            result = np.append(result, 0)
        elif probability[1] > thresh:
            result = np.append(result, 1)
        elif probability[0] != probability[1]:
            result = np.append(result, -1)
        else:
            msg = f"Something is wrong with the probability at index: {i}!"
            raise ValueError(msg)
    return result


def predict_proba(
        classifier: ProbaClassifier,
        test_x: NDArray | list[Verse],
        test_labels: NDArray,
        threshold: float = 0.5,
    ) -> ProbaPredictions:
    """Predict on the data with the classifier and return some statistics.

    Args:
        classifier: any object that has a method `.predict_proba()`.
        test_x: inputs to the classifier
        test_labels: correct labels (gold references) for the inputs. This is
            not used to predict probabilities, but is used to calculate
            prediction accuracy etc.
        threshold: the threshold for probability to be counted as a label for
            a particular class.

    Returns:
        a :class:`classifier.result_utils.ProbaPredictions` instance.
    """
    y_pred_proba = classifier.predict_proba(test_x)
    # convert the list of probas to a list of labels
    y_pred = convert(y_pred_proba, threshold)

    return ProbaPredictions(test_x, y_pred, y_pred_proba,
                            test_labels)
