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

"""Utility functions to make predictions with a src.classifier."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.classifier.result_utils import (
    Predictions,
    ProbaPredictions,
    Verse,
)
from src.classifier.wrappers import Classifier, ProbaClassifier


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
    y_pred = src.classifier.predict(test_samples)
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
        threshold: float = 0.5,
    ) -> ProbaPredictions:
    """Predict on the data with the classifier and return the probabilities.

    Args:
        classifier: any object that has a method `.predict_proba()`.
        test_x: inputs to the classifier
        test_labels: correct labels (gold references) for the inputs. This is
            not used to predict probabilities, but is used to calculate
            prediction accuracy etc.
        threshold: the threshold for probability to be counted as a label for
            a particular class.

    Returns:
        a :class:`src.classifier.result_utils.ProbaPredictions` instance.
    """
    y_pred_proba = src.classifier.predict_proba(test_x)
    # convert the list of probas to a list of labels
    y_pred = convert(y_pred_proba, threshold)

    return ProbaPredictions(test_x, y_pred, y_pred_proba)
