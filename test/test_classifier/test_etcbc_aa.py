from unittest.mock import patch

from src.classifier.dataset_skeleton import LoadedDataset
from src.classifier.etcbc_aa_eval import (
    csvify_etcbc,
    etcbc_eval_classifier,
    remove_non_chars,
    remove_proper_nouns,
)
from src.classifier.fname_utils import SavefileName
from src.classifier.result_utils import Verse
from src.classifier.wrappers import BoWEstimator


def test_csvify_etcbc(etcbc_verse: Verse) -> None:
    """Test the csvify_etcbc function."""
    samples = [etcbc_verse]
    probas = [[0.8, 0.2]]
    correct_labels = [0]

    # Test with correct labels
    result = csvify_etcbc(samples, probas, correct_labels)
    expected = (
        '"Book","Reference","Probability for Jewish","Probability for Christian",'
        + '"ETCBC Transliteration","ܐܠܦܒܝܬ ܣܘܪܝܝܐ","Correct Label"\n'
        + '"Genesis","Genesis Chapter 01 Verse 01",0.8000,0.2000,'
        + "BRCJT BR> >LH> JT CMJ> WJT >R<>,ܒܪܫܝܬ ܒܪܐ ܐܠܗܐ ܝܬ ܫܡܝܐ ܘܝܬ ܐܪܥܐ,0\n"
    )
    assert result == expected

    # Test without correct labels
    result_no_labels = csvify_etcbc(samples, probas)
    expected_no_labels = (
        '"Book","Reference","Probability for Jewish","Probability for Christian",'
        + '"ETCBC Transliteration","ܐܠܦܒܝܬ ܣܘܪܝܝܐ"\n'
        + '"Genesis","Genesis Chapter 01 Verse 01",0.8000,0.2000,'
        + "BRCJT BR> >LH> JT CMJ> WJT >R<>,ܒܪܫܝܬ ܒܪܐ ܐܠܗܐ ܝܬ ܫܡܝܐ ܘܝܬ ܐܪܥܐ\n"
    )
    assert result_no_labels == expected_no_labels


def test_remove_proper_nouns(etcbc_verse: Verse) -> None:
    """Test the remove_proper_nouns function."""
    result = remove_proper_nouns(etcbc_verse)
    assert isinstance(result, Verse)
    assert "BRCJT" in result.get_translit_words()
    assert ">BRHM" not in result.get_translit_words()  # Example proper noun


def test_remove_non_chars() -> None:
    """Test the remove_non_chars function."""
    verse = ["BRCJT", "BR>", ">LH>", "JT\u0308", "CMJ>", "WJT\u0307", ">R<>"]
    result = remove_non_chars(verse)
    expected = ["BRCJT", "BR>", ">LH>", "JT", "CMJ>", "WJT", ">R<>"]
    assert result == expected


@patch("src.classifier.etcbc_aa_eval.eval_and_save")
def test_etcbc_eval_classifier(
    mock_eval_and_save,
    mnb_classifier: BoWEstimator,
    loaded_etcbc: LoadedDataset,
) -> None:
    """Test the etcbc_eval_classifier function."""
    save_fname = SavefileName("ETCBC", "mnb", "csv")

    # Call the function
    etcbc_eval_classifier(mnb_classifier, loaded_etcbc, save_fname)

    # Check that the classifier has been trained
    assert mnb_classifier.vocabs is not None
    mock_eval_and_save.assert_called_once_with(
        mnb_classifier,
        loaded_etcbc,
        csvify_etcbc,
        save_fname,
        out_dir="./src/classifier/out/",
    )


@patch("src.classifier.etcbc_aa_eval.eval_and_save")
def test_etcbc_eval_classifier_with_mapping(
    mock_eval_and_save,
    mnb_classifier: BoWEstimator,
    loaded_etcbc: LoadedDataset,
) -> None:
    """Test the etcbc_eval_classifier function with a mapping function."""
    save_fname = SavefileName("ETCBC", "mnb", "csv")

    # Call the function with a mapping function
    etcbc_eval_classifier(
        mnb_classifier,
        loaded_etcbc,
        save_fname,
        func_to_map=remove_proper_nouns,
        map_to_both=True,
    )

    # Check that the classifier has been trained
    assert mnb_classifier.vocabs is not None

    mock_eval_and_save.assert_called_once_with(
        mnb_classifier,
        loaded_etcbc,
        csvify_etcbc,
        save_fname,
        out_dir="./src/classifier/out/",
    )
