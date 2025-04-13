import pytest

from src.classifier.dataset_skeleton import DataSplit
from src.classifier.result_utils import Verse


class TestDataSplit:
    def test_init(self,
                  cal_verse: Verse,
                  cal_one_word_verse: Verse,
                  cal_romans_verse: Verse,
                  cal_ds: DataSplit) -> None:
        # check initialised values
        assert (cal_ds.num_classes == 2)
        # the initialised ``verses`` should be a two-dimensional list,
        # each index of the first dimension representing the class label (0, 1)
        assert (len(cal_ds.verses) == cal_ds.num_classes)
        assert (cal_ds.verses[0] == [cal_verse, cal_one_word_verse])
        assert (cal_ds.verses[1] == [cal_romans_verse])

    def test_get_samples(self,
                         etcbc_chr_verses: list[Verse],
                         cal_romans_verse: Verse) -> None:
        local_ds = DataSplit(etcbc_chr_verses, [cal_romans_verse])
        comp_list = etcbc_chr_verses.copy()
        comp_list.extend([cal_romans_verse])
        assert (local_ds.get_samples() == comp_list)
        assert (local_ds.get_samples(0) == etcbc_chr_verses)
        assert (local_ds.get_samples(1) == [cal_romans_verse])
        # ensure that this func raises ValueError upon invalid arguments
        with pytest.raises(ValueError, match=(
                    "Integer label \\d cannot be generated. "
                   + "They are not in the expected labels for this system."
                   + " Please refer to the documentation for: "
                   + "datasets.DataSplit")):
            local_ds.get_samples(2)

        with pytest.raises(ValueError, match=(
                    "Integer label -\\d cannot be generated. "
                   + "They are not in the expected labels for this system."
                   + " Please refer to the documentation for: "
                   + "datasets.DataSplit")):
            local_ds.get_samples(-1)

    def test_get_labels(self, cal_ds: DataSplit) -> None:
        assert (cal_ds.get_labels() == [0, 0, 1])
        assert (cal_ds.get_labels(0) == [0, 0])
        assert (cal_ds.get_labels(1) == [1])

    def test_generate_syriac_hf(self,
                                cal_ds: DataSplit,
                                cal_romans_verse: Verse,
                                etcbc_ds: DataSplit,
                                etcbc_chr_verses: list[Verse],
                                etcbc_acts_verse: Verse,
                                etcbc_cor1_verse: Verse
                                ) -> None:
        # check it works with no params
        hf_lines = list(cal_ds.generate_syriac_hf())
        assert (hf_lines[0] == {"label": 0, "text": ""})
        assert (hf_lines[1] == {"label": 0, "text": ""})
        assert (hf_lines[2] == {"label": 1, "text": " ".join(
            cal_romans_verse.get_syriac_words())})
        # check it works for specific labels
        # for OT
        hf_lines_0 = list(etcbc_ds.generate_syriac_hf(0))
        assert (hf_lines_0[0] == {"label": 0, "text": " ".join(
            etcbc_chr_verses[0].get_syriac_words())})
        assert (hf_lines_0[1] == {"label": 0, "text": " ".join(
            etcbc_chr_verses[1].get_syriac_words())})
        assert (hf_lines_0[2] == {"label": 0, "text": " ".join(
            etcbc_chr_verses[2].get_syriac_words())})
        # for NT
        hf_lines_1 = list(etcbc_ds.generate_syriac_hf(1))
        assert (hf_lines_1[0] == {"label": 1, "text": " ".join(
            etcbc_acts_verse.get_syriac_words())})
        assert (hf_lines_1[1] == {"label": 1, "text": " ".join(
            etcbc_cor1_verse.get_syriac_words())})
