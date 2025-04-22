# BSD 2-Clause License
#
# Copyright (c) 2025, Ryosuke Nagata
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Utility functions to make predictions with a classifier."""

import logging
from math import isclose

import numpy as np
from numpy.typing import NDArray

from src.classifier.result_utils import (
    Predictions,
    ProbaPredictions,
    Verse,
)
from src.classifier.wrappers import Classifier, ProbaClassifier

logger = logging.getLogger(__name__)


def predict(
    clf: Classifier,
    test_samples: NDArray | list[Verse],
    test_labels: NDArray | list[int],
) -> Predictions:
    """Predict on the data with a classifier and get some simple statistics.

    Args:
        clf: any object that has a method `.predict()`.
        test_samples: inputs to the classifier
        test_labels: correct labels (gold references) for the inputs. This is
            not used to predict probabilities, but is used to calculate
            prediction accuracy etc.

    Returns:
        A :class:`Prediction` class instance.
    """
    y_pred = clf.predict(test_samples)
    return Predictions(test_samples, y_pred, test_labels)


def convert(probas: NDArray | list[list[float]], thresh: float) -> NDArray:
    """Convert list of probabilities to a list of labels.

    Args:
        probas: list of predicted probabilities.
        thresh: threshold to decide if a probability prediction should be
            labelled as an instance of the class.

    Returns:
        0 if prediction for label 0 is over the threshold,
        1 if prediction for label 1 is over the threshold,
        -1 if probabilities for no label exceeds the threshold, or if more than
            one class is have probabilities beyond the threshold.

    Raises:
        ValueError: if the sum of probabilities for a sample is not 1.0.
    """
    if thresh < 0.5:
        msg = (
            "The threshold is below 0.5, which leads to many cases with "
            + "unknown classification labels since multiple classes"
            + "can easily have probabilities over the threshold."
        )
        logger.warning(msg)

    result = np.empty(0, dtype=int)
    num_classes = len(probas[0])  # number of classes to classify samples into

    for i in range(len(probas)):
        probability = probas[i]

        if not isclose(np.sum(probability), 1.0):
            # raise if the sum of probability is not 1
            msg = f"Something is wrong with the probability at index: {i}!"
            raise ValueError(msg)

        proba_class = -1
        # flag to detect if the sample at i is already assigned to a class with
        # a probability exceeding the thresh.
        already_classified = False
        for j in range(num_classes):
            if probability[j] > thresh and already_classified:
                msg = (
                    f"Probability for the sample at {i} has more than one "
                    + "class that has probability over the threshold "
                    + f"{thresh}. This is converted to label unknown (-1)."
                )
                logger.info(msg)
                proba_class = -1
            elif probability[j] > thresh:
                # if the probability of class j exceeds thresh
                proba_class = j
                already_classified = True

        result = np.append(result, proba_class)
    return result


def predict_proba(
    clf: ProbaClassifier,
    test_x: NDArray | list[Verse],
    test_y: NDArray | list[int] | None = None,
    *,
    threshold: float = 0.5,
) -> ProbaPredictions:
    """Predict on the data with the classifier and return the probabilities.

    Args:
        clf: any object that has a method `.predict_proba()`.
        test_x: inputs to the classifier
        test_y: correct labels of the test data
        test_labels: correct labels (gold references) for the inputs. This is
            not used to predict probabilities, but is used to calculate
            prediction accuracy etc.
        threshold: the threshold for probability to be counted as a label for
            a particular class.

    Returns:
        a :class:`src.classifier.result_utils.ProbaPredictions` instance.
    """
    y_pred_proba = clf.predict_proba(test_x)
    # convert the list of probas to a list of labels
    y_pred = convert(y_pred_proba, threshold)

    return ProbaPredictions(test_x, y_pred, y_pred_proba, test_y)
