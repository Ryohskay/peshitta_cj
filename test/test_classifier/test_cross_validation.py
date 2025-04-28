from math import isclose
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch
from sklearn.naive_bayes import MultinomialNB

from src.classifier.cross_validation import (
    cross_validate_clf,
    find_best_clf,
)
from src.classifier.dataset_skeleton import LoadedDataset
from src.classifier.fname_utils import SavefileName


@pytest.fixture
def mock_cross_validate() -> MagicMock:
    n_fold = 3
    max_n = 3
    mock_ret = []
    for i in range(1, max_n + 1):
        val = 0.1 * i
        mock_ret.append(
            (
                [val] * n_fold,
                [[val, val] * n_fold],
                [[val, val] * n_fold],
                [[val, val] * n_fold],
            )
        )
    assert len(mock_ret) == max_n
    mock_cross_validate = MagicMock()
    mock_cross_validate.side_effect = mock_ret
    return mock_cross_validate


def test_cross_validate_clf(
    monkeypatch: MonkeyPatch,
    loaded_etcbc: LoadedDataset,
    mock_cross_validate: MagicMock,
):
    """Test the cross_validate_clf function."""
    max_n = 3
    # Mock the cross_validate function
    monkeypatch.setattr(
        "src.classifier.cross_validation.cross_validate",
        mock_cross_validate,
    )

    # test
    algo = MultinomialNB()
    max_idx, max_fscore, configs = cross_validate_clf(
        algo, n_window_max=3, ds=loaded_etcbc
    )

    # Assertions
    assert len(configs) == max_n
    assert max_idx == 2
    assert isclose(max_fscore, 0.3)  # Average of [0.3] * 3
    assert configs[0]["n_window"] == 1
    assert configs[1]["n_window"] == 2
    assert configs[2]["n_window"] == 3


def test_find_best_clf(
    monkeypatch: MonkeyPatch,
    loaded_etcbc: LoadedDataset,
    etcbc_base_savefile: SavefileName,
):
    """Test the find_best_clf function."""
    # Patch the cross_validate_clf function
    mock_cross_validate_clf = MagicMock(
        return_value=(
            1,
            0.85,
            [{"n_window": 1}, {"n_window": 2}, {"n_window": 3}],
        )
    )
    monkeypatch.setattr(
        "src.classifier.cross_validation.cross_validate_clf",
        mock_cross_validate_clf,
    )
    mock_run_etcbc_clf = MagicMock()
    monkeypatch.setattr(
        "src.classifier.cross_validation.run_etcbc_clf",
        mock_run_etcbc_clf,
    )
    # Call the function
    max_c_fscore, max_w_fscore = find_best_clf(
        clf_alias="mnb",
        n_window_max=3,
        ds=loaded_etcbc,
        base_fname=etcbc_base_savefile,
    )

    # Assertions
    assert max_c_fscore == 0.85
    assert max_w_fscore == 0.85
    mock_cross_validate_clf.assert_called()


# @patch("src.classifier.cross_validation.cross_validate_clf")
# @patch("src.classifier.cross_validation.run_cal_clf")
# def test_find_best_clf_cal(
#     mock_run_cal_clf,
#     mock_cross_validate_clf,
#     mock_loaded_dataset,
#     mock_savefile_name,
# ):
#     """Test the find_best_clf function for CAL data."""
#     # Mock the cross_validate_clf function
#     mock_cross_validate_clf.return_value = (
#         1,
#         0.9,
#         [{"n_window": 1}, {"n_window": 2}],
#     )

#     # Modify the SavefileName to represent CAL data
#     mock_savefile_name.origin = "CAL"

#     # Call the function
#     max_c_fscore, max_w_fscore = find_best_clf(
#         clf_alias="mnb",
#         n_window_max=3,
#         ds=mock_loaded_dataset,
#         base_fname=mock_savefile_name,
#     )

#     # Assertions
#     assert max_c_fscore == 0.9
#     assert max_w_fscore == 0.9
#     mock_run_cal_clf.assert_called()
