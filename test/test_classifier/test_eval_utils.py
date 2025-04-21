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
    ResultStats,
    Verse,
)
from src.classifier.wrappers import BoWEstimator


def test_mislabel_stats_raises(proba_preds_no_correct: ProbaPredictions) -> None:
    with pytest.raises(ValueError, match=("The attr `preds.correct_labels` is "
        + "empty. Correct labels are required to find mislabelled verses.")):
        mislabel_stats(proba_preds_no_correct)


def test_mislabel_stats(proba_preds_mislab: ProbaPredictions) -> None:
    result = mislabel_stats(proba_preds_mislab)
    assert isinstance(result, Mislabels)
    assert len(result.mislabels) == 4


def test_metricise() -> None:
    y_true = [0, 1, 0, 1]
    y_all = [0, 1, 1, 0]
    y_probas = [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]]
    accuracy, precision, recall, fbeta, sta = metricise(y_true, y_all, y_probas)
    assert isinstance(accuracy, float)
    assert accuracy == 0.5
    assert len(precision) == 2
    assert len(recall) == 2
    assert len(fbeta) == 2
    assert isinstance(sta, ResultStats)


def test_split_list() -> None:
    # when the list is not divisible by parts
    lis = [1, 2, 3, 4, 5]
    result = split_list(lis, parts=2)
    assert len(result) == 2
    assert result[0] == [1, 2]
    assert result[1] == [3, 4, 5]
    # when the list is diviesible by parts
    lis = [1, 2, 3, 4, 5, 6]
    result = split_list(lis, parts=3)
    assert len(result) == 3
    assert result[0] == [1, 2]
    assert result[1] == [3, 4]
    assert result[2] == [5, 6]

def test_split_list_raises() -> None:
    lis = [1, 2]
    with pytest.raises(ValueError,
                       match="Cannot divide a list of length 2 into 5 parts!"):
        split_list(lis, parts=5)


def test_evaluate_classifier(mnb_classifier: BoWEstimator,
                             loaded_etcbc: LoadedDataset) -> None:
    mnb_classifier.fit(loaded_etcbc.train.get_samples(),
                       loaded_etcbc.train.get_labels())
    result = evaluate_classifier(mnb_classifier, loaded_etcbc.test,
                                    plot=False, threshold=0.5)
    assert len(result) == 5
    assert isinstance(result[0], ProbaPredictions)
    assert isinstance(result[1], ProbaPredictions)
    assert result[2] is None
    assert isinstance(result[3], Mislabels)
    assert isinstance(result[4], ResultStats)



def test_cross_validate(
        mnb_classifier: BoWEstimator,
        etcbc_chr_verses: list[Verse],
        etcbc_verse: Verse,
        full_data_verse: Verse,
        cal_one_word_verse: Verse,
        cal_romans_verse: Verse,
        etcbc_acts_verse: Verse,
        etcbc_cor1_verse: Verse,
    ) -> None:
    mock_clf = mnb_classifier
    training_x = etcbc_chr_verses
    training_x.extend([
                        etcbc_verse,
                        full_data_verse,
                        cal_one_word_verse,
                        cal_romans_verse,
                        etcbc_acts_verse,
                        etcbc_cor1_verse
                       ])
    training_y = [0, 0, 0, 0, 0, 0, 1, 1, 1]

    cross_validate(mock_clf, training_x, training_y, fold=3, threshold=0.5)
