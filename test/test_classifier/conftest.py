"""pytest.fixture objects to reuse among tests"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB

from src.classifier.dataset_skeleton import DataSplit, LoadedDataset
from src.classifier.fitting_utils import identity
from src.classifier.fname_utils import SavefileName
from src.classifier.load_cal import BookData
from src.classifier.result_utils import (
    Mislabels,
    ProbaPredictions,
    Verse,
)
from src.classifier.wrappers import BoWEstimator, Classifier


# Predictions & ProbaPredictions
@pytest.fixture
def proba_preds(
    cal_verse: Verse, etcbc_verse: Verse, full_data_verse: Verse
) -> ProbaPredictions:
    """Return a ProbaPredictions instance with verses from Genesis."""
    return ProbaPredictions(
        [cal_verse, etcbc_verse, full_data_verse],
        [0, 0, 1],
        [[0.8, 0.2], [0.7, 0.3], [0.4, 0.6]],
        [0, 0, 0],
    )


@pytest.fixture
def proba_preds_no_correct(
    etcbc_verse: Verse, etcbc_acts_verse: Verse
) -> ProbaPredictions:
    """Return a ProbaPredictions instance with no correct labels."""
    return ProbaPredictions(
        [etcbc_verse, etcbc_acts_verse], [0, 1], [[0.7, 0.3], [0.4, 0.6]]
    )


@pytest.fixture
def proba_preds_mislab(
    etcbc_chr_verses: list[Verse],
    etcbc_acts_verse: Verse,
    etcbc_cor1_verse: Verse,
) -> ProbaPredictions:
    """Return a ProbaPredictions instance with several mislabels."""
    verses = etcbc_chr_verses
    verses.extend([etcbc_acts_verse, etcbc_cor1_verse])
    return ProbaPredictions(
        verses,
        [1, 1, 0, 0, 0],
        [[0.11, 0.89], [0.33, 0.67], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
        [0, 0, 0, 1, 1],
    )


@pytest.fixture
def proba_multi_preds(
    cal_verse: Verse, etcbc_chr_verse: Verse
) -> ProbaPredictions:
    """Return a ProbaPredictions instance with verses of Gen and 1 Chr."""
    return ProbaPredictions(
        [cal_verse, etcbc_chr_verse],
        [0, 1],
        [[0.8, 0.2], [3.2e-10, 0.9]],
        [0, 0],
    )


@pytest.fixture
def proba_pred_zero(
    etcbc_verse: Verse,
    etcbc_acts_verse: Verse,
    etcbc_cor1_verse: Verse,
) -> ProbaPredictions:
    """Mock ProbaPredictions object containing zero proba for a verse."""
    return ProbaPredictions(
        [etcbc_verse, etcbc_acts_verse, etcbc_cor1_verse],
        [0, 1, 0],
        [[0.7, 0.3], [0.4, 0.6], [1.0, 0.0]],
        [0, 1, 1],
    )


# Mislabels
@pytest.fixture
def misls(etcbc_chr_verses: list[Verse]) -> Mislabels:
    """Return a Mislabels instance with multiple verses."""
    return Mislabels(
        [1, 1, 1],
        [0, 0, 0],
        etcbc_chr_verses,
        [[1.8e-10, 0.9], [0.667, 0.333], [0.401, 0.60]],
    )


@pytest.fixture
def mislabels_both(
    etcbc_verse: Verse,
    etcbc_cor1_verse: Verse,
) -> Mislabels:
    """Mock Mislabels with OT and NT verses."""
    return Mislabels(
        [1, 0],
        [0, 1],
        [etcbc_verse, etcbc_cor1_verse],
        [[0.3, 0.7], [0.6, 0.4]],
    )


# DataSplit
@pytest.fixture
def cal_ds(
    cal_verse: Verse, cal_one_word_verse: Verse, cal_romans_verse: Verse
) -> DataSplit:
    """Return a DataSplit object with CAL data."""
    return DataSplit([cal_verse, cal_one_word_verse], [cal_romans_verse])


@pytest.fixture
def etcbc_ds(
    etcbc_chr_verses: list[Verse],
    etcbc_acts_verse: Verse,
    etcbc_cor1_verse: Verse,
) -> DataSplit:
    """Return a DataSplit object with ETCBC data."""
    return DataSplit(etcbc_chr_verses, [etcbc_acts_verse, etcbc_cor1_verse])


@pytest.fixture
def loaded_etcbc(
    etcbc_chr_verses: list[Verse],
    etcbc_cor1_verse: Verse,
    etcbc_verse: Verse,
    etcbc_acts_verse: Verse,
    full_data_verse: Verse,
) -> LoadedDataset:
    """Return a LoadedDataset object with ETCBC data."""
    return LoadedDataset(
        etcbc_chr_verses,
        [etcbc_cor1_verse],
        [etcbc_verse],
        [etcbc_acts_verse],
        [full_data_verse],
    )


@pytest.fixture
def loaded_cal(
    cal_verse: Verse,
    cal_one_word_verse: Verse,
    cal_romans_verse: Verse,
    full_data_verse: Verse,
) -> LoadedDataset:
    """Return a LoadedDataset object with CAL data."""
    return LoadedDataset(
        [cal_verse, cal_one_word_verse],
        [cal_romans_verse],
        [cal_verse],
        [cal_romans_verse],
        [full_data_verse],
    )


# Classifier
@pytest.fixture
def lr_classifier() -> Classifier:
    """Mock LinearRegression classifier."""
    # ArgumentType can be ignored here since LinearRegression inherits from the
    # BaseEstimator class
    return Classifier(LinearRegression())


@pytest.fixture
def mnb_classifier() -> BoWEstimator:
    """Mock MultinomialNB classifier."""
    return BoWEstimator(MultinomialNB(), identity, n=3)


@pytest.fixture
def mock_bow_clf():
    """Mock BoWEstimator object."""
    mock_clf = MagicMock(spec=BoWEstimator)
    mock_clf.fit.return_value = None
    mock_clf.predict_proba.return_value = [[0.8, 0.2], [0.6, 0.4]]
    return mock_clf


@pytest.fixture
def mock_formatter(
    samples: list[Verse] | NDArray[Verse],
    probas: list[list[float]] | NDArray[np.float64],
    correct_labels: list[int] | NDArray[np.int64] | None = None,
) -> str:
    """Mock formatter implementing FileFormatterProto."""
    if correct_labels is not None:
        return "A,B,C,D,E"
    # else
    return "A,B,C,D"


@pytest.fixture
def mock_predict_proba(proba_preds: ProbaPredictions) -> MagicMock:
    return MagicMock(return_value=proba_preds)


@pytest.fixture
def mock_eval_and_save():
    """Mock eval_and_save function."""
    return MagicMock(return_value=None)


# SavefileName
@pytest.fixture
def cal_base_savefile() -> SavefileName:
    return SavefileName(origin="CAL", classifier_alias="mnb", file_ext="csv")


@pytest.fixture
def etcbc_base_savefile() -> SavefileName:
    return SavefileName(origin="ETCBC", classifier_alias="svc", file_ext="json")


# BookData
@pytest.fixture
def cal_book_data_gen() -> BookData:
    """Mock BookData dictionary."""
    return {
        "book_title": "Genesis",
        "verse_refs": ["Genesis Chapter 01 Verse 01"],
        "lemmatised_verses": [
            ["br$yt", "br)", ")lh)", "yt", "$my)", "w_", "yt", ")r()"]
        ],
        "lemma_annotations": [
            [
                "noun sg. abs. or construct",
                "verb G",
                "noun sg. emphatic",
                "p01",
                "noun pl. emphatic",
                "c",
                "p01",
                "noun sg. emphatic",
            ],
        ],
    }


@pytest.fixture
def cal_book_data_est() -> BookData:
    return {
        "book_title": "Esther",
        "verse_refs": [
            "Esther Chapter 00 Verse 00",
            "Esther Chapter 01 Verse 01",
            "Esther Chapter 01 Verse 02",
        ],
        "lemmatised_verses": [
            ["ktb", "d_", ")styr"],
            [
                "w_",
                "hwy",
                "b_",
                "ywm",
                "d_",
                ")x$yr$",
                "hw",
                "br",
                "d_",
                ")x$yr$",
                "d_",
                "mlk",
                "mn",
                "hwd",
                "w_",
                "(dm)",
                "l_",
                "kw$",
                "(l",
                "m))",
                "w_",
                "(sryn",
                "mdynh",
            ],
            [
                "b_",
                "ywm",
                "hnwn",
                "kd",
                "ytb",
                "hwy",
                "mlk",
                ")x$yr$",
                "(l",
                "kwrsy",
                "d_",
                "mlkw",
                "d_",
                "b_",
                "$w$n",
                "byrh",
            ],
        ],
        "lemma_annotations": [
            ["noun sg. emphatic", "p = d_ p --> dy p", "PN Personal name"],
            [
                "c",
                "verb G",
                "p02",
                "noun pl. emphatic",
                "p = d_ p --> dy p",
                "PN Personal name",
                "P01",
                "noun sg. emphatic",
                "p = d_ p --> dy p",
                "PN Personal name",
                "c = d_ c --> dy c",
                "verb C",
                "p01",
                "GN Geographic name",
                "c",
                "c = (dm) c --> (dm) p",
                "p03",
                "GN Geographic name",
                "p01",
                "n01 = m)) b --> m)h b",
                "c",
                "n01",
                "noun pl. absolute",
            ],
            [
                "p02",
                "noun pl. emphatic",
                "P01",
                "c",
                "verb G",
                "verb G",
                "noun sg. emphatic",
                "PN Personal name",
                "p01",
                "noun sg. emphatic",
                "p = d_ p --> dy p",
                "noun sg. emphatic",
                "c = d_ c --> dy c",
                "p02",
                "GN Geographic name",
                "noun sg. emphatic",
            ],
        ],
    }


@pytest.fixture
def cal_books_data(cal_book_data_gen, cal_book_data_est) -> list[BookData]:
    return [cal_book_data_gen, cal_book_data_est]
