import pytest
from pathlib import Path
from src.ui_web.load_predictions import load_preds, BookVerses
from src.classifier.fname_utils import FnameExtraOpts, SavefileName
from src.classifier.result_utils import Verse


csv_dir_p = Path("assets/classifier_results")

savefile_name_etcbc = SavefileName(
    origin="ETCBC",
    classifier_alias="mnb",
    file_ext="csv")
savefile_name_etcbc.set_ngram_opts(n=3, is_n_gram=True, is_bow=True, is_char_level=True)
savefile_name_etcbc.add_extra_opts(
    [
        FnameExtraOpts.REMOVE_DIACRITICS
    ]
)
etcbc_jewish = savefile_name_etcbc.copy()
etcbc_jewish.set_scope("Jewish")
etcbc_christian = savefile_name_etcbc.copy()
etcbc_christian.set_scope("Christian")

savefile_name_cal = SavefileName(
    origin="CAL",
    classifier_alias="mnb",
    file_ext="csv")
savefile_name_cal.set_ngram_opts(n=3, is_n_gram=True, is_bow=True, is_char_level=True)
savefile_name_cal.add_extra_opts(
    [
        FnameExtraOpts.REMOVE_PROPN,
        FnameExtraOpts.REMOVE_FROM_BOTH,
    ]
)
cal_jewish = savefile_name_cal.copy()
cal_jewish.set_scope("Jewish")
cal_christian = savefile_name_cal.copy()
cal_christian.set_scope("Christian")

def test_load_preds_etcbc():
    """Test the load_preds function for ETCBC data."""
    # Call the function
    result = load_preds(etcbc_jewish, load_dir=csv_dir_p)

    # Assertions
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["book_name"] == "Deuteronomy"
    assert len(result[0]["verses"]) == 2
    assert isinstance(result[0]["verses"][0], Verse)
    assert result[0]["verses"][0].get_translit_words() == [
        "WHLJN",
        'PT"GM>',
        "D>MR",
        "MWC>",
        "LKLH",
        ">JSRJL",
        "B<BR>",
        "DJWRDNN",
        "BMDBR>",
        "B<RB>",
        "LWQBL",
        "SWP",
        "BJT",
        "PRN",
        "WBJT",
        "TPL",
        "WLBNN",
        "WXYRWT",
        "WRZHB",
    ]
    assert result[0]["verse_probas"] == [[1.0, 0.0], [0.9999, 0.0001]]


def test_load_preds_cal():
    """Test the load_preds function for CAL data."""
    # Call the function
    result = load_preds(cal_christian, load_dir=csv_dir_p)

    # Assertions
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["book_name"] == "Acts"
    assert len(result[0]["verses"]) == 3
    assert isinstance(result[0]["verses"][0], Verse)
    assert result[0]["verses"][0].get_translit_words() == ["ktb", "qdmy", "ktb", ")w", "(l", "kl", ")ylyn", "d_", "$ry", "mry", "m$yx", "l_", "(bd", "w_", "l_", ")lp"]
    assert result[0]["verse_probas"] == [[0.0000,1.0000], [0.0000,1.0000], [0.0000,1.0000]]