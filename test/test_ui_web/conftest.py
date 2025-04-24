from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.classifier.fname_utils import FnameExtraOpts, SavefileName
from src.classifier.result_utils import (
    ResultStats,
    ThresholdStats,
    Verse,
    jsonify_result_stats,
)
from src.ui_web.load_book_probas import BookProbas
from src.ui_web.load_clf_stats import JsonifiedSummaryDict
from src.ui_web.load_predictions import BookVerses
from src.ui_web.select_load_classifier import (
    ClassifierConfig,
    ClassifierResultsModel,
    ResultFilesIndex,
)


@pytest.fixture
def mock_load_dir():
    return Path("./assets/classifier_results/")


@pytest.fixture
def mock_save_fname():
    """Fixture to provide a mock SavefileName object."""
    fname = SavefileName(origin="CAL", classifier_alias="mnb", file_ext="csv")
    fname.set_ngram_opts(n=3, is_n_gram=True, is_bow=True, is_char_level=True)
    fname.add_extra_opts(
        [
            FnameExtraOpts.REMOVE_PROPN,
            FnameExtraOpts.REMOVE_UNDERSCORES,
            FnameExtraOpts.REMOVE_FROM_BOTH,
        ]
    )
    fname.mark_special_file(is_total_proba=True)
    return fname


@pytest.fixture
def mock_classifier_config():
    """Fixture to provide a mock ClassifierConfig object."""
    return ClassifierConfig(
        name="mnb",
        origin="ETCBC",
        is_n_gram=True,
        n=3,
        is_bow=True,
        is_char_level=True,
        extra_opts=[FnameExtraOpts.REMOVE_DIACRITICS],
    )


@pytest.fixture
def mock_result_files_index():
    """Fixture to provide a mock ResultFilesIndex object."""
    mock_index = MagicMock(spec=ResultFilesIndex)
    mock_index.match_files_by_config.return_value = [
        SavefileName(origin="ETCBC", classifier_alias="mnb", file_ext="csv"),
        SavefileName(origin="ETCBC", classifier_alias="mnb", file_ext="json"),
    ]
    return mock_index


@pytest.fixture
def mock_result_files_index_assets(mock_load_dir: Path) -> ResultFilesIndex:
    """Fixture to provide a mock ResultFilesIndex object."""
    return ResultFilesIndex(mock_load_dir)


@pytest.fixture
def mock_cal_book_probas() -> BookProbas:
    """Mock BookProbas object from CAL data."""
    bp = BookProbas("CAL")
    bp.add_book_proba("Genesis", [0.8, 0.2])
    bp.add_book_proba("Esther", [0.6, 0.4])
    return bp


@pytest.fixture
def mock_jsonified_summary_dict(
    result_stats_1: ResultStats,
) -> JsonifiedSummaryDict:
    """Mock JsonifiedSummaryDict object."""
    return JsonifiedSummaryDict(
        n_gram_form="word",
        n=3,
        total_n_grams_parsed=1000,
        top_ten_in_training=[
            (("word1", "word2"), 10),
            (("word2", "word3"), 20),
        ],
        test_mislabel_percent={"Genesis": 0.1, "Esther": 0.2},
        metrics=jsonify_result_stats(result_stats_1),
    )


@pytest.fixture
def mock_list_book_verses(
    cal_verse: Verse,
    cal_romans_verse: Verse,
    etcbc_chr_verses: list[Verse]
) -> list[BookVerses]:
    return [
        {
            "book_name": "Genesis",
            "verses": [cal_verse] * 5,
            "verse_probas": [[0.8, 0.2]] * 5,
        },
        {
            "book_name": "Romans",
            "verses": [cal_romans_verse] * 5,
            "verse_probas": [[0.6, 0.4]] * 5,
        },
        {
            "book_name": "Chronicles_1",
            "verses": etcbc_chr_verses,
            "verse_probas": [[0.7, 0.3]] * len(etcbc_chr_verses),
        },
    ]


@pytest.fixture
def mock_cal_clf_conf() -> ClassifierConfig:
    return ClassifierConfig(
        name="mnb",
        origin="CAL",
        is_n_gram=True,
        n=3,
        is_bow=True,
        is_char_level=False,
    )


@pytest.fixture
def mock_classifier_results_model(
    mock_cal_clf_conf: ClassifierConfig,
    mock_result_files_index_assets: ResultFilesIndex,
    mock_load_dir: Path,
) -> ClassifierResultsModel:
    return ClassifierResultsModel(
        mock_cal_clf_conf, mock_result_files_index_assets, mock_load_dir
    )
