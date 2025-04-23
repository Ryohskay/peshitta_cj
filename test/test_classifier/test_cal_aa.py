import pytest
from unittest.mock import MagicMock, patch
from src.classifier.cal_aa_eval import (
    csvify_cal,
    remove_underscores,
    remove_enclitics,
    remove_proper_nouns,
    cal_eval_classifier,
)
from src.classifier.result_utils import Verse
from src.classifier.wrappers import BoWEstimator
from src.classifier.fname_utils import SavefileName
from src.classifier.dataset_skeleton import LoadedDataset


def test_csvify_cal(cal_verse: Verse) -> None:
    """Test the csvify_cal function."""
    samples = [cal_verse]
    probas = [[0.8, 0.2]]
    correct_labels = [0]

    # Test with correct labels
    result = csvify_cal(samples, probas, correct_labels)
    expected = (
        "Book,Reference,\"Probability for Jewish\",\"Probability for Christian\",Leammatised Verse,Correct Label\n"
        "Genesis,Genesis Chapter 01 Verse 01,0.8000,0.2000,br$yt br) )lh) yt $my) w_ yt )r(),0\n"
    )
    assert result == expected

    # Test without correct labels
    result_no_labels = csvify_cal(samples, probas)
    expected_no_labels = (
        "Book,Reference,\"Probability for Jewish\",\"Probability for Christian\",Leammatised Verse\n"
        "Genesis,Genesis Chapter 01 Verse 01,0.8000,0.2000,br$yt br) )lh) yt $my) w_ yt )r()\n"
    )
    assert result_no_labels == expected_no_labels


def test_remove_underscores() -> None:
    """Test the remove_underscores function."""
    words = ["br$yt", "w_", "yt", ")r()"]
    result = remove_underscores(words)
    expected = ["br$yt", "w", "yt", ")r()"]
    assert result == expected


def test_remove_enclitics(cal_verse: Verse) -> None:
    """Test the remove_enclitics function."""
    # test with a verse containing proclitic conjunction and prepositions
    result = remove_enclitics(cal_verse)
    assert isinstance(result, Verse)
    assert result.get_translit_words() == [
            "br$yt",
            "br)",
            ")lh)",
            "$my)",
            ")r()"]
    assert result.get_annotations() == [
            "noun sg. abs. or construct",
            "verb G",
            "noun sg. emphatic",
            "noun pl. emphatic",
            "noun sg. emphatic",
        ]

    # Test with no enclitics, this should leave the verse intact
    verse_no_enclitics = Verse(
        "Genesis",
        "Genesis Chapter 01 Verse 01",
        ["br$yt", "br)", ")lh)"],
        ["noun sg. abs. or construct", "verb G", "noun sg. emphatic"],
        origin="CAL",
    )
    result_no_enclitics = remove_enclitics(verse_no_enclitics)
    assert verse_no_enclitics == result_no_enclitics

    # Test with a verse containing only enclitics
    enclitics_verse = Verse(
        "Genesis",
        "Genesis Chapter 01 Verse 01",
        ["w_", "yt"],
        words_annotations=["c", "p01"],
        origin="CAL",
    )
    result_enclitic_verse = remove_enclitics(enclitics_verse)
    assert result_enclitic_verse is None


def test_remove_proper_nouns(cal_verse: Verse, cal_romans_verse: Verse) -> None:
    """Test the remove_proper_nouns function."""
    # Test with a verse without proper nouns, this should leave the verse intact
    result = remove_proper_nouns(cal_verse)
    assert isinstance(result, Verse)
    assert result.get_translit_words() == ["br$yt", "br)", ")lh)", "yt", "$my)", "w_", "yt", ")r()"]
    assert result.get_annotations() == [
        "noun sg. abs. or construct",
        "verb G",
        "noun sg. emphatic",
        "p01",
        "noun pl. emphatic",
        "c",
        "p01",
        "noun sg. emphatic",
    ]

    # Test with a verse containing a proper noun
    result_rem_propn = remove_proper_nouns(cal_romans_verse)
    assert result_rem_propn is not None
    assert result_rem_propn.book == "Romans"
    assert result_rem_propn.get_syriac_words() == [
            "ܥܰܒܼܕܿܳܐ",
            "ܕ",
            "ܡܫܺܝܚܳܐ",
            "ܩܰܪܝܳܐ",
            "ܘܰ",
            "ܫܠܺܝܚܳܐ",
            "ܕܶ",
            "ܐܬܼܦܿܪܶܫ",
            "ܠܶ",
            "ܐܘܰܢܓܿܶܠܺܝܳܘܢ",
            "ܕܰ",
            "ܐܠܳܗܳܐ",
        ]
    assert result_rem_propn.get_translit_words() == [
            "(bd",
            "d_",
            "m$yx",
            "qry",
            "w_",
            "$lyx",
            "d_",
            "pr$",
            "l_",
            ")wnglywn",
            "d_",
            ")lh",
        ]
    assert result_rem_propn.get_annotations() == [
            "noun sg. emphatic",
            "p",
            "noun sg. emphatic",
            "verb G",
            "c",
            "noun sg. emphatic",
            "c",
            "Verb Gt",
            "p03",
            "noun sg. abs. or construct",
            "p",
            "noun sg. emphatic",
        ]

    # test with a verse containing only proper nouns
    propn_verse = Verse("Romans", "Romans Chapter 01 Verse 01",
            [
            "pwlws",
            "y$w("
        ],
        syriac_words=[
            "ܦܿܰܘܠܳܘܣ",
            "ܝܶܫܽܘܥ",
        ],
        words_annotations=[
            "PN Personal Name",
            "PN Personal Name"
        ]
        )
    propn_verse_result = remove_proper_nouns(propn_verse)
    assert propn_verse_result is None


@patch("src.classifier.cal_aa_eval.eval_and_save")
def test_cal_eval_classifier(mock_eval_and_save, mnb_classifier: BoWEstimator, loaded_cal: LoadedDataset) -> None:
    """Test the cal_eval_classifier function."""
    save_fname = SavefileName("CAL", "mnb", "csv")

    # Call the function
    cal_eval_classifier(mnb_classifier, loaded_cal, save_fname)

    # Assertions
    # check that the classifier has been trained
    assert mnb_classifier.vocabs is not None
    mock_eval_and_save.assert_called_once_with(
        mnb_classifier,
        loaded_cal,
        csvify_cal,
        save_fname,
        out_dir="./src/classifier/out/",
    )
