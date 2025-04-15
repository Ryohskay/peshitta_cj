"""Test classes and their methods in result_utils.py."""
import re

import pytest

from src.classifier.result_utils import (
    Mislabels,
    PeshittaWord,
    Predictions,
    ProbaPredictions,
    Verse,
)


class TestPeshittaWord:
    def test_eq(self, cal_word: PeshittaWord,
                etcbc_word: PeshittaWord,
                words: object) -> None:
        # totally different stuff
        assert (cal_word != etcbc_word)
        # identical objects
        etcbc_words = words.copy(2)  # type: ignore[reportAttributeAccessIssue]
        assert (etcbc_words[0] == etcbc_words[1])
        # PeshittaWord obj vs. str
        assert (cal_word == ")lh")
        assert (etcbc_word == "ܒܢ̈ܝܗ̇")
        assert (cal_word != "br)$yt")
        assert (etcbc_word != "BRCJT")


class TestVerse:
    def test_init_raises(self,
                         cal_word: PeshittaWord,
                         etcbc_word: PeshittaWord
                         ) -> None:
        # Make sure that __init__ raises error(s) upon invalid arguments
        with pytest.raises(ValueError,
                           match=(re.escape("book_name (None), verse_ref (None),"
                                            " and translit_words (None) given: "
                                            "they cannot be None!"))
                           ):
            Verse(book_name=None, verse_ref=None,
                        translit_words=None,
                        syriac_words=None,
                        words_annotations=None)

        with pytest.raises(ValueError,
                           match=(re.escape("The number of words in transliteration 1 and"
                              + " in Syriac script 11 do not match."))
                          ):
            Verse("Chronicles_1",
                  "1 Chronicles Chapter 01 Verse 33",
                  [etcbc_word.translit],
                  syriac_words=["ܘܒ̈ܢܝ", "ܡܕܝܢ", "ܥܦܐ", "ܘܐܥܦܪ", "ܘܚܢܘܟ", "ܘܐܒܝܕܥ", "ܘܐܠܪܥܐ", "ܗܠܝܢ", "ܟܠܗܘܢ", "ܒܢ̈ܝܗ̇", "ܕܩܢܛܘܪܐ"],
                  origin="ETCBC"
                  )

        with pytest.raises(ValueError,
                           match=(re.escape("The number of words in transliteration 7 and"
                              + " the number of annotations 1  do not match."))
                           ):
            Verse("Genesis",
                  "Genesis 01 Chapter 01 Verse 01",
                  translit_words=["br$yt", "br)", ")lh)",
                                  "yt", "$my)", "wyt", ")r()"],
                  words_annotations=[cal_word.annots],
                  origin="CAL")

    def test_full_arg_init(self,
                           full_data_verse: Verse,
                           full_data_words: list[PeshittaWord],
                           genesis_1_1: str
                           ) -> None:
        # Test that initialisation with all arguments work
        assert (full_data_verse.book == "Genesis")
        assert (full_data_verse.reference == genesis_1_1)
        for i in range(len(full_data_words)):
            assert (full_data_words[i] == full_data_verse.words[i])

    def test_etcbc_verse_init(self,
                              etcbc_verse: Verse,
                              etcbc_words: list[PeshittaWord],
                              genesis_1_1: str
                              ) -> None:
        # test the case where transliteration and syriac are given
        assert (etcbc_verse.book == "Genesis")
        assert (etcbc_verse.reference == genesis_1_1)
        for i in range(len(etcbc_words)):
            assert (etcbc_verse.words[i] == etcbc_words[i])
            # check that the attrs not provided are initialised with empty strs
            assert (not etcbc_verse.words[i].annots)

    def test_cal_verse_init(self,
                            cal_verse: Verse,
                            cal_words: list[PeshittaWord],
                            genesis_1_1: str
                            ) -> None:
        # test the case where transliteration and syriac are given
        assert (cal_verse.book == "Genesis")
        assert (cal_verse.reference == genesis_1_1)
        for i in range(len(cal_words)):
            assert (cal_verse.words[i] == cal_words[i])
            # check that the attrs not provided are initialised with empty strs
            assert (not cal_verse.words[i].syriac)

    def test_translit_init(self,
                           cal_word: PeshittaWord,
                           genesis_1_1: str
                           ) -> None:
        # test the case where only transliteration is provided.
        vrs = Verse(book_name="Genesis",
                    verse_ref=genesis_1_1,
                    translit_words=[cal_word.translit]
                    )
        assert (vrs.book == "Genesis")
        assert (vrs.reference == genesis_1_1)
        for i in range(len(vrs.words)):
            assert (vrs.get_translit_words()[i] == [cal_word.translit][i])

    def test_get_translit_words(self,
                                cal_one_word_verse: Verse,
                                cal_word: PeshittaWord,
                                etcbc_verse: Verse,
                                etcbc_words: list[PeshittaWord]) -> None:
        assert (cal_one_word_verse.get_translit_words() == [cal_word.translit])
        assert (etcbc_verse.get_translit_words() ==
                    [w.translit for w in etcbc_words])

    def test_get_syriac_words(self,
                              cal_syriacs: list[str],
                              cal_verse: Verse,
                              full_data_verse: Verse) -> None:
        assert (full_data_verse.get_syriac_words() == cal_syriacs)
        # Check that this returns an empty string if word.syriac not given
        empty_str_cnts = 0
        for syriac in cal_verse.get_syriac_words():
            if not syriac:  # if the string is empty
                empty_str_cnts += 1
        assert (len(cal_verse) == empty_str_cnts)

    def test_eq(self,
                cal_verse: Verse,
                cal_one_word_verse: Verse,
                full_data_verse: Verse,
                etcbc_verse: Verse,
                cal_word: PeshittaWord,
                cal_syriacs: list[str],
                genesis_1_1: str
                ) -> None:
        # test equality comparator
        # totally different stuff
        assert (etcbc_verse != cal_one_word_verse)
        # Verse vs. list[str]
        assert (cal_verse != cal_syriacs)
        # identical objects
        vrs = Verse(book_name="Genesis", verse_ref=genesis_1_1,
                    translit_words=[cal_word.translit],
                    words_annotations=[cal_word.annots]
                    )
        assert (vrs == cal_one_word_verse)
        # same verse, different word partitioning
        assert (full_data_verse != etcbc_verse)
        # same verse, with and without syriac alphabets
        assert (full_data_verse == cal_verse)
        # Verse vs. list[PeshittaWord]
        assert (cal_one_word_verse == [cal_word])

    def test_str(self,
                 full_data_verse: Verse,
                 full_data_words: list[PeshittaWord],
                 cal_verse: Verse,
                 cal_translits: list[str],
                 genesis_1_1: str) -> None:
        # test string conversion
        transliterations = " ".join([w.translit
                                     for w in full_data_words])
        comp_str = f"{genesis_1_1} | {transliterations}"
        assert (str(full_data_verse) == comp_str)
        comp_str = f"{genesis_1_1} | {' '.join(cal_translits)}"
        assert (str(cal_verse) == comp_str)

    def test_len(self,
                 cal_one_word_verse: Verse,
                 etcbc_chr_verse: Verse
                 ) -> None:
        # test length response
        assert (len(cal_one_word_verse) == 1)
        assert (len(etcbc_chr_verse) == 11)

    def test_get_words_in_mode(self,
                               full_data_verse: Verse,
                               cal_translits: list[str],
                               cal_verse: Verse,
                               cal_syriacs: list[str]) -> None:
        # it should return a list of transliteration by default
        assert (full_data_verse.get_words_in_mode() == cal_translits)
        # mode=1 for transliteration
        assert (full_data_verse.get_words_in_mode(mode=1) == cal_translits)
        # mode=2 for syriac
        assert (full_data_verse.get_words_in_mode(mode=2) == cal_syriacs)
        # Check that this returns an empty string if word.syriac not given
        empty_str_cnts = 0
        for syriac in cal_verse.get_syriac_words():
            if not syriac:  # if the string is empty
                empty_str_cnts += 1
        assert (len(cal_verse) == empty_str_cnts)
        # check it raises exception for other undefined modes
        with pytest.raises(ValueError,
                           match=r"Argument `mode` must be 1 or 2, but \d"
                           + " was found."):
            assert (full_data_verse.get_words_in_mode(mode=0) == cal_translits)
        with pytest.raises(ValueError,
                           match=r"Argument `mode` must be 1 or 2, but \d"
                           + " was found."):
            assert (full_data_verse.get_words_in_mode(mode=0) == cal_translits)

    def test_get_annotations(self,
                             full_data_verse: Verse,
                             etcbc_verse: Verse,
                             cal_annots: list[str]) -> None:
        assert (full_data_verse.get_annotations() == cal_annots)
        # Check that this returns an empty string if word.annots not given
        empty_str_cnts = 0
        for annot in etcbc_verse.get_annotations():
            if not annot:  # if the string is empty
                empty_str_cnts += 1
        assert (len(etcbc_verse) == empty_str_cnts)


class TestPredictions:
    def test_init(self,
                  cal_verse: Verse,
                  etcbc_verse: Verse,
                  full_data_verse: Verse,
                  ) -> None:
        # Check exception is raised when lengths of arguemnt lists don't match
        with pytest.raises(ValueError, match=(re.escape(
            "length of provided lists/arrays for samples and predictions "
            + "do not match."))):
            assert Predictions([full_data_verse], [1, 0])

        with pytest.raises(ValueError, match=(re.escape(
            "length of provided lists/arrays for samples and correct"
              + " labels do not match."))):
            assert Predictions([full_data_verse], [1], [0, 0])

        # Check that normal initialisation works
        preds = Predictions([full_data_verse], [1], [0])
        assert (preds.samples == [full_data_verse])
        assert (preds.predictions == [1])
        assert (preds.correct_labels == [0])
        n_preds = Predictions([cal_verse, etcbc_verse, full_data_verse],
                              [1, 1, 0],
                              [0, 0, 0])
        assert (n_preds.samples == [cal_verse, etcbc_verse, full_data_verse])
        assert (n_preds.predictions == [1, 1, 0])
        assert (n_preds.correct_labels == [0, 0, 0])


class TestProbaPredictions:
    def test_init(self,
                  cal_verse: Verse,
                  etcbc_verse: Verse,
                  full_data_verse: Verse,
                  ) -> None:
        with pytest.raises(ValueError, match=(re.escape(
            r"lengths of probas 2 and samples 3 do not match!"))):
            ProbaPredictions([cal_verse, etcbc_verse, full_data_verse],
                              [0, 0, 1],
                              [[0.8, 0.2], [0.7, 0.3]],
                             [0, 0, 0])
        probas = ProbaPredictions([cal_verse, etcbc_verse, full_data_verse],
                          [0, 0, 1],
                          [[0.8, 0.2], [0.7, 0.3], [0.4, 0.6]],
                         [0, 0, 0])
        assert (probas.samples == [cal_verse, etcbc_verse, full_data_verse])
        assert (probas.predictions == [0, 0, 1])
        assert (probas._probas[0] == [0.8, 0.2])
        assert (probas._probas[1] == [0.7, 0.3])
        assert (probas._probas[2] == [0.4, 0.6])
        assert (probas.correct_labels == [0, 0, 0])

    def test_get_probas(self, proba_preds: ProbaPredictions) -> None:
        assert (proba_preds.get_probas() ==
                [[0.8, 0.2], [0.7, 0.3], [0.4, 0.6]])

    def test_get_total_probas(self,
                              proba_preds: ProbaPredictions,
                              proba_multi_preds: ProbaPredictions,
                              ) -> None:
        # Check for the case with one book
        aj = 0.8 * 0.7 * 0.4
        ac = 0.2 * 0.3 * 0.6
        assert (proba_preds.get_total_probas() == {"Genesis": [aj, ac]})
        # Check for the case with multiple books
        assert (proba_multi_preds.get_total_probas() ==
                {"Genesis": [0.8, 0.2], "Chronicles_1": [3.2e-10, 0.9]})


class TestMislabels:
    def test_init(self,
                  etcbc_chr_verses: list[Verse]) -> None:
        probas_misl = Mislabels([1, 1, 1], [0, 0, 0],
                etcbc_chr_verses,
                                   probas=[[0.1, 0.9], [0.7, 0.3],
                                           [0.33, 0.67]],
                            )
        assert (probas_misl.mislabels == [1, 1, 1])
        assert (probas_misl.correct_labels == [0, 0, 0])
        assert (probas_misl.verses == etcbc_chr_verses)
        assert (probas_misl.probas == [[0.1, 0.9], [0.7, 0.3], [0.33, 0.67]])

    def test_len(self, misls: Mislabels) -> None:
        assert (len(misls) == 3)
