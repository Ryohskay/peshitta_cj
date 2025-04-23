from pathlib import Path

from src.classifier.fname_utils import FnameExtraOpts, SavefileName
from src.ui_web.select_load_classifier import (
    ClassifierConfig,
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
