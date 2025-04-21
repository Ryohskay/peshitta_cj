import pytest

from src.classifier.fname_utils import FnameExtraOpts, SavefileName


class TestSavefileName:
    def test_init(self):
        fname = SavefileName(
            origin="CAL", classifier_alias="mnb", file_ext="csv"
        )
        assert fname.origin == "CAL"
        assert fname.classifier == "mnb"
        assert fname.ext == "csv"
        # non-alnum chars must be removed
        fname_etc = SavefileName(
            origin="CAL",
            classifier_alias=("סיווג בייסיאני" + "\u05be" + "נאיבי"),
            file_ext="csv",
        )
        assert fname_etc.origin == "CAL"
        assert fname_etc.ext == "csv"
        assert fname_etc.classifier == "סיווג_בייסיאני_נאיבי"

    def test_init_raises(self):
        with pytest.raises(ValueError, match="Data origin is empty!"):
            SavefileName(
                origin="",  # type: ignore[reportArgumentType]
                classifier_alias="mnb",
                file_ext="csv",
            )

        with pytest.raises(ValueError, match="Unknown data origin: INVALID"):
            # pass an invalid origin
            SavefileName(
                origin="INVALID",  # type: ignore[reportArgumentType]
                classifier_alias="mnb",
                file_ext="csv",
            )

        with pytest.raises(ValueError, match="Classifier alias is empty!"):
            SavefileName(origin="CAL", classifier_alias="", file_ext="csv")

        with pytest.raises(ValueError, match="File extension is empty!"):
            SavefileName(origin="CAL", classifier_alias="mnb", file_ext="")

    def test_is_same_classifier(self):
        fname1 = SavefileName(
            origin="CAL", classifier_alias="mnb", file_ext="csv"
        )
        fname2 = SavefileName(
            origin="CAL", classifier_alias="mnb", file_ext="csv"
        )
        fname3 = SavefileName(
            origin="ETCBC", classifier_alias="mnb", file_ext="csv"
        )
        assert fname1.is_same_classifier(fname2)
        assert not fname1.is_same_classifier(fname3)

    def test_as_path(self, cal_base_savefile: SavefileName):
        assert str(cal_base_savefile.as_path()) == cal_base_savefile.get_fname()

    def test_set_ngram_opts(self, cal_base_savefile: SavefileName):
        # with is_ngram = False, n is ignored
        # other values should be kept intact
        cal_base_savefile.set_ngram_opts(n=9, is_n_gram=False)
        assert not cal_base_savefile.is_n_gram
        assert cal_base_savefile.n == 0
        assert not cal_base_savefile.is_char_level
        assert not cal_base_savefile.is_bow
        # with default options
        cal_base_savefile.set_ngram_opts(n=3)
        assert cal_base_savefile.n == 3
        assert cal_base_savefile.is_char_level
        assert cal_base_savefile.is_n_gram
        assert cal_base_savefile.is_bow
        # overwriting
        cal_base_savefile.set_ngram_opts(n=6, is_char_level=False, is_bow=False)
        assert cal_base_savefile.n == 6
        assert not cal_base_savefile.is_char_level
        assert cal_base_savefile.is_n_gram
        assert not cal_base_savefile.is_bow

    def test_set_scope(self, cal_base_savefile: SavefileName):
        # check that the scope name is normalised
        cal_base_savefile.set_scope("Jewish")
        assert cal_base_savefile.scope == "jewish"

    def test_set_scope_raises(self, cal_base_savefile: SavefileName):
        with pytest.raises(
            ValueError, match="Invalid scope or label name: INVALID"
        ):
            cal_base_savefile.set_scope("INVALID")

    def test_add_extra_opts(
        self, cal_base_savefile: SavefileName, etcbc_base_savefile: SavefileName
    ):
        # Check with a single opt
        cal_base_savefile.add_extra_opts([FnameExtraOpts.REMOVE_DIACRITICS])
        assert cal_base_savefile.extra_opts[FnameExtraOpts.REMOVE_DIACRITICS]
        # Multiple opts consecutively added
        etcbc_base_savefile.add_extra_opts([FnameExtraOpts.IS_ERRONEOUS])
        etcbc_base_savefile.add_extra_opts(
            [FnameExtraOpts.REMOVE_DIACRITICS, FnameExtraOpts.REMOVE_FROM_BOTH]
        )
        assert etcbc_base_savefile.extra_opts[FnameExtraOpts.IS_ERRONEOUS]
        assert etcbc_base_savefile.extra_opts[FnameExtraOpts.REMOVE_DIACRITICS]
        assert etcbc_base_savefile.extra_opts[FnameExtraOpts.REMOVE_FROM_BOTH]

    def test_add_extra_opts_raises(self, cal_base_savefile: SavefileName):
        with pytest.raises(ValueError, match="Invalid option: INVALID"):
            cal_base_savefile.add_extra_opts(["INVALID"])  # type: ignore[reportArgumentType]

    def test_mark_special_file(self, cal_base_savefile: SavefileName):
        cal_base_savefile.mark_special_file(is_mislabel=True)
        assert cal_base_savefile.is_mislabel

    def test_mark_special_file_raises(self):
        fname = SavefileName(
            origin="CAL", classifier_alias="mnb", file_ext="csv"
        )
        with pytest.raises(
            ValueError,
            match="Mislabelled verse is undefined for production data",
        ):
            # impossible combination of config opts
            fname.mark_special_file(is_mislabel=True, is_prod=True)

    def test_get_fname(self, cal_base_savefile: SavefileName):
        cal_base_savefile.set_ngram_opts(
            n=3, is_char_level=True, is_n_gram=True, is_bow=True
        )
        cal_base_savefile.set_scope("Jewish")
        cal_base_savefile.add_extra_opts([FnameExtraOpts.REMOVE_UNDERSCORES])
        cal_base_savefile.mark_special_file(is_mislabel=True)
        expected = "cal_mnb_char_3gram_bow_jewish_no_uscore_mislabels.csv"
        assert cal_base_savefile.get_fname() == expected
        # the order of extra options should be automatically aligned
        cal_base_savefile.ext = "json"
        cal_base_savefile.add_extra_opts(
            [FnameExtraOpts.IS_ERRONEOUS, FnameExtraOpts.REMOVE_DIACRITICS]
        )
        cal_base_savefile.mark_special_file(is_prod=True)
        expected_new = expected.split("_")[:6]
        expected_new.extend(
            [
                "no",
                "diacritics",
                "no",
                "uscore",
                "ERRONEOUS",
                "prediction",
                "all",
            ]
        )
        expected = "PRODUCTION_" + "_".join(expected_new) + ".json"
        assert cal_base_savefile.get_fname() == expected
