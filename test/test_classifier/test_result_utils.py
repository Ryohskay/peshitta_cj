"""Test classes and their methods in result_utils.py."""
import pytest

import re
from classifier.result_utils import PeshittaWord, Verse


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

    def test_eq(self,
                cal_verse: Verse,
                cal_one_word_verse: Verse,
                full_data_verse: Verse,
                etcbc_verse: Verse,
                cal_word: PeshittaWord,
                cal_syriacs: list[str],
                genesis_1_1: str
                ):
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

    def test_get_translit_words(self):
        pass

    def test_get_syriac_words(self):
        pass

    def test_get_words_in_mode(self):
        pass

    def test_get_annotations(self):
        pass


class TestPredictions:
    def test_init(self):
        pass


class TestProbaPredictions:
    def test_init(self):
        pass

    def test_get_probas(self):
        pass

    def test_get_total_probas(self):
        pass

    def test_save_to_file(self):
        pass


class TestMislabels:
    def test_init(self):
        pass

    def test_len(self):
        pass

    def test_save_to_file(self):
        pass
