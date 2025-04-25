from json import load
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch

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


def test_configure_fname_opts(monkeypatch: MonkeyPatch):
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
    # mock all setter methods that define a classifier spec
    for attr in dir(SavefileName):
        if (
            callable(getattr(save_fname, attr))
            and attr.startswith("set")
            and attr != "set_scope"
        ):
            # for each method of SavefileName class
            monkeypatch.setattr(save_fname, attr, MagicMock(name=attr))
    # call the function to test
    configure_fname_opts(save_fname, config)
    # check that all setter methods of SavefileName, specifying a classifier,
    # are called once and only once by the configure_fname_opts function.
    for attr in dir(SavefileName):
        if (
            callable(getattr(save_fname, attr))
            and attr.startswith("set")
            and attr != "set_scope"
        ):
            # for each method of SavefileName class
            getattr(save_fname, attr).assert_called_once()


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


def test_verify_load(
    mock_classifier_config: ClassifierConfig,
    mock_result_files_index: ResultFilesIndex,
    mock_load_dir: Path,
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
    mock_classifier_config: ClassifierConfig,
    mock_result_files_index: ResultFilesIndex,
    mock_load_dir: Path,
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


class TestClassifierResultsModel:
    def test_classifier_results_model_init(
        self,
        mock_cal_clf_conf: ClassifierConfig,
        mock_load_dir: Path,
        mock_result_files_index_assets: ResultFilesIndex,
    ):
        """Test the ClassifierResultsModel class."""
        model = ClassifierResultsModel(
            mock_cal_clf_conf, mock_result_files_index_assets, mock_load_dir
        )
        assert model.config == mock_cal_clf_conf
        assert model.index == mock_result_files_index_assets
        assert model.load_dir == mock_load_dir
        assert not model._is_loaded

    def test_load_results(
        self,
        mock_cal_book_probas: BookProbas,
        mock_classifier_results_model: ClassifierResultsModel,
        loaded_clf_stats: JsonifiedSummaryDict,
    ):
        """Test the load_results method."""
        # Call the method
        mock_classifier_results_model.load_results()

        assert mock_classifier_results_model._is_loaded
        assert mock_classifier_results_model.book_probas is not None
        assert (
            mock_classifier_results_model.book_probas.book_proba_dict
            == mock_cal_book_probas.book_proba_dict
        )
        assert mock_classifier_results_model.results_summary == loaded_clf_stats

    def test_load_results_book_verses(
        self,
        loaded_book_verses: list[BookVerses],
        mock_result_files_index_assets: ResultFilesIndex,
        mock_load_dir: Path,
    ):
        """Test that load_results method correctly loads BookVerses dicts."""
        config = ClassifierConfig(
            "mnb",
            "CAL",
            is_n_gram=True,
            n=3,
            is_bow=True,
            is_char_level=True,
            extra_opts=[
                FnameExtraOpts.REMOVE_PROPN,
                FnameExtraOpts.REMOVE_FROM_BOTH,
            ],
        )
        model = ClassifierResultsModel(
            config, mock_result_files_index_assets, mock_load_dir
        )
        model.load_results()
        loaded_book_names = [v["book_name"] for v in model.book_verses]
        loaded_verses = [v["verses"] for v in model.book_verses]
        loaded_probas = [v["verse_probas"] for v in model.book_verses]

        assert len(model.book_verses) > 0
        assert len(loaded_book_verses) == len(model.book_verses)
        for i in range(len(loaded_book_verses)):
            assert loaded_book_verses[i]["book_name"] in loaded_book_names
            corresp_idx = loaded_book_names.index(loaded_book_verses[i]["book_name"])
            assert loaded_book_verses[i]["verses"] == loaded_verses[corresp_idx]
            assert loaded_book_verses[i]["verse_probas"] == loaded_probas[
                corresp_idx
            ]

    def test_verify_load(
        self, mock_classifier_results_model: ClassifierResultsModel
    ):
        """Test the _verify_load method."""
        # Verify that an exception is raised if data is not loaded
        with pytest.raises(
            ValueError, match=r"Data not loaded. Call `load_results\(\)` first."
        ):
            mock_classifier_results_model._verify_load()

        # Load the data and verify no exception is raised
        mock_classifier_results_model._is_loaded = True
        mock_classifier_results_model._verify_load()
