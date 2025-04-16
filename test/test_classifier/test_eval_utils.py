"""Tests for the eval_utils module."""
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.classifier.dataset_skeleton import LoadedDataset
from src.classifier.eval_utils import (
    cross_validate,
    eval_and_save,
    evaluate_classifier,
    metricise,
    mislabel_stats,
    split_list,
)
from src.classifier.result_utils import (
    FileFormatterProto,
    Mislabels,
    ProbaPredictions,
    Verse,
)
from src.classifier.wrappers import BoWEstimator


def test_mislabel_stats_raises(proba_predictions: ProbaPredictions) -> None:
    inputs = np.empty(0)
    y_correct = np.array([0, 1])
    with pytest.raises(ValueError, match=("Cannot measure the size of `inputs`!"
               + " It seems like the argument `inputs` is empty.")):
        mislabel_stats(inputs, y_correct, proba_predictions)

def test_mislabel_stats(proba_predictions: ProbaPredictions) -> None:
    inputs = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_correct = np.array([0, 1, 0, 1])
    result = mislabel_stats(inputs, y_correct, proba_predictions)
    assert isinstance(result, Mislabels)
    assert len(result.mislabels) == 2


def test_metricise() -> None:
    y_true = [0, 1, 0, 1]
    y_all = [0, 1, 1, 0]
    y_probas = [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]]
    accuracy, precision, recall, fbeta = metricise(y_true, y_all, y_probas)
    assert isinstance(accuracy, float)
    assert len(precision) == 2
    assert len(recall) == 2
    assert len(fbeta) == 2


def test_split_list() -> None:
    lis = [1, 2, 3, 4, 5]
    result = split_list(lis, parts=2)
    assert len(result) == 2
    assert result[0] == [1, 2]
    assert result[1] == [3, 4, 5]


def test_split_list_raises() -> None:
    lis = [1, 2]
    with pytest.raises(ValueError,
                       match="Cannot divide a list of length 2 into 5 parts!"):
        split_list(lis, parts=5)


def test_evaluate_classifier(mnb_classifier: BoWEstimator,
                             loaded_etcbc: LoadedDataset) -> None:
    result = evaluate_classifier(mnb_classifier, loaded_etcbc,
                                 plot=False, threshold=0.5)
    assert len(result) == 4
    assert isinstance(result[0], ProbaPredictions)
    assert isinstance(result[2], Mislabels)


def test_eval_and_save(
        tmp_path: Path,
        mnb_classifier: BoWEstimator,
        loaded_etcbc: LoadedDataset,
        mock_formatter: FileFormatterProto) -> None:

    result = eval_and_save(
        mnb_classifier,
        loaded_etcbc,
        mock_formatter,
        out_dir=str((tmp_path / "out/").resolve()),
        save_file_prefix="test_",
        save_file_suffix="_suffix",
        save_file_ext=".txt",
        threshold=0.5,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_cross_validate() -> None:
    mock_clf = MagicMock(spec=BoWEstimator)
    training_x = [MagicMock(spec=Verse) for i in range(10)]
    training_y = [0, 1] * 5

    cross_validate(mock_clf, training_x, training_y, fold=2, threshold=0.5)
    mock_clf.fit.assert_called()
