from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.classifier.fname_utils import FnameExtraOpts, SavefileName
from src.ui_web.load_book_probas import BookProbas
from src.ui_web.load_clf_stats import JsonifiedSummaryDict
from src.ui_web.load_predictions import BookVerses
from src.ui_web.select_load_classifier import (
    ClassifierConfig,
    ClassifierResultsModel,
    ResultFilesIndex,
    configure_fname_opts,
    parse_fname,
)


def test_classifier_config_init():
    """Test the ClassifierConfig dataclass."""
    config = ClassifierConfig(
        name="mnb",
        origin="CAL",
        is_n_gram=True,
        n=3,
        is_bow=True,
        is_char_level=False,
        extra_opts=[FnameExtraOpts.REMOVE_DIACRITICS],
    )
    assert config.name == "mnb"
    assert config.origin == "CAL"
    assert config.is_n_gram
    assert config.n == 3
    assert config.is_bow
    assert not config.is_char_level
    assert config.extra_opts == [FnameExtraOpts.REMOVE_DIACRITICS]


def test_configure_fname_opts():
    """Test the configure_fname_opts function."""
    save_fname = SavefileName(
        origin="CAL", classifier_alias="mnb", file_ext="csv"
    )
    config = ClassifierConfig(
        name="mnb",
        origin="CAL",
        is_n_gram=True,
        n=3,
        is_bow=True,
        is_char_level=False,
        extra_opts=[FnameExtraOpts.REMOVE_DIACRITICS],
    )
    configured_fname = configure_fname_opts(save_fname, config)
    assert configured_fname.n == 3
    assert configured_fname.is_n_gram
    assert configured_fname.is_bow
    assert not configured_fname.is_char_level
    assert configured_fname.origin == "CAL"
    assert configured_fname.classifier == "mnb"
    assert configured_fname.ext == "csv"
    assert not configured_fname.is_mislabel
    assert not configured_fname.is_total_proba
    assert not configured_fname.is_clf_summary
    assert FnameExtraOpts.REMOVE_DIACRITICS in configured_fname.extra_opts


def test_parse_fname():
    """Test the parse_fname function."""
    fname = "etcbc_mnb_char_3gram_bow_jewish_no_diacritics_mislabels.csv"
    parsed_fname = parse_fname(fname)
    assert parsed_fname.origin == "ETCBC"
    assert parsed_fname.classifier == "mnb"
    assert parsed_fname.is_char_level
    assert parsed_fname.n == 3
    assert parsed_fname.is_bow
    assert parsed_fname.is_mislabel
    assert FnameExtraOpts.REMOVE_DIACRITICS in parsed_fname.extra_opts
    # production file
    fname = "PRODUCTION_cal_mnb_char_3gram_bow_jewish_no_diacritics.csv"
    parsed_fname = parse_fname(fname)
    assert parsed_fname.origin == "CAL"
    assert parsed_fname.classifier == "mnb"
    assert parsed_fname.is_prod


class TestResultFilesIndex:
    def test_result_files_index(self):
        """Test the ResultFilesIndex class."""
        assets_dir = Path("./assets/classifier_results/")
        index = ResultFilesIndex(assets_dir)
        assert len(index) == len(list(assets_dir.iterdir()))
        for i in range(len(index)):
            f = index.files[i]
            assert f.origin in {"ETCBC", "CAL"}
            assert f.classifier == "mnb"
            assert f.is_n_gram
            assert f.n == 3
            assert f.is_bow
            assert index.file_paths[i].name == f.get_fname()
            if "mislabels" in index.file_paths[i].name:
                assert f.is_mislabel
        assert index.files[0].origin == "CAL"

    def test_match_files_by_config(self):
        """Test the match_files_by_config method."""
        config = ClassifierConfig(
            name="mnb",
            origin="CAL",
            is_n_gram=True,
            n=3,
            is_bow=True,
            is_char_level=False,
        )
        index = ResultFilesIndex("./assets/classifier_results/")

        matched_files = index.match_files_by_config(config)
        assert len(matched_files) == 2
        for file in matched_files:
            assert file.origin == config.origin
            assert file.classifier == config.name
            assert file.n == config.n

        # check if it works with files containing extra options
        config = ClassifierConfig(
            name="mnb",
            origin="ETCBC",
            is_n_gram=True,
            n=3,
            is_bow=True,
            is_char_level=True,
            extra_opts=[FnameExtraOpts.REMOVE_DIACRITICS],
        )
        matched_files = index.match_files_by_config(config)
        assert len(matched_files) == 3


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
def mock_load_dir(tmp_path):
    """Fixture to provide a temporary directory for loading files."""
    return tmp_path


@patch("src.ui_web.select_load_classifier.load_book_probas")
@patch("src.ui_web.select_load_classifier.load_clf_stats")
@patch("src.ui_web.select_load_classifier.load_preds")
def test_load_results(
    mock_load_preds,
    mock_load_clf_stats,
    mock_load_book_probas,
    mock_classifier_config,
    mock_result_files_index,
    mock_load_dir,
):
    """Test the load_results method."""
    # Mock the loading functions
    mock_load_book_probas.return_value = MagicMock(spec=BookProbas)
    mock_load_clf_stats.return_value = MagicMock(spec=JsonifiedSummaryDict)
    mock_load_preds.return_value = [MagicMock(spec=BookVerses)]

    # Create the ClassifierResultsModel instance
    model = ClassifierResultsModel(
        config=mock_classifier_config,
        result_index=mock_result_files_index,
        load_dir=mock_load_dir,
    )

    # Call the load_results method
    model.load_results()

    # Assertions
    mock_result_files_index.match_files_by_config.assert_called_once_with(
        mock_classifier_config
    )
    mock_load_book_probas.assert_called_once()
    mock_load_clf_stats.assert_called_once()
    mock_load_preds.assert_called_once()
    assert model.book_probas is not None
    assert model.results_summary is not None
    assert len(model.book_verses) == 1
    assert model._is_loaded is True
    assert model.results_summary is not None



def test_verify_load(
    mock_classifier_config, mock_result_files_index, mock_load_dir
):
    """Test the _verify_load method."""
    model = ClassifierResultsModel(
        config=mock_classifier_config,
        result_index=mock_result_files_index,
        load_dir=mock_load_dir,
    )

    # Verify that an exception is raised if data is not loaded
    with pytest.raises(
        ValueError, match=r"Data not loaded. Call `load_results\(\)` first."
    ):
        model._verify_load()

    # Load the data and verify no exception is raised
    model._is_loaded = True
    model._verify_load()


def test_get_book_verses(
    mock_classifier_config, mock_result_files_index, mock_load_dir
):
    """Test the get_book_verses method."""
    model = ClassifierResultsModel(
        config=mock_classifier_config,
        result_index=mock_result_files_index,
        load_dir=mock_load_dir,
    )

    # Mock the load_results method
    model.load_results = MagicMock()
    model.book_verses = [MagicMock(spec=BookVerses)]
    model._is_loaded = True

    # Call the method and verify the result
    result = model.get_book_verses()
    assert len(result) == 1
    assert isinstance(result[0], dict)

    # Verify that load_results is not called again
    model.load_results.assert_not_called()
