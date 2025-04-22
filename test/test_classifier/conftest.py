"""pytest.fixture objects to reuse among tests"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB

from src.classifier.dataset_skeleton import DataSplit, LoadedDataset
from src.classifier.fitting_utils import identity
from src.classifier.fname_utils import SavefileName
from src.classifier.load_cal import BookData
from src.classifier.result_utils import (
    Mislabels,
    PeshittaWord,
    ProbaPredictions,
    ResultStats,
    ThresholdStats,
    Verse,
)
from src.classifier.wrappers import BoWEstimator, Classifier


# Verse & PeshittaWord
@pytest.fixture
def genesis_1_1() -> str:
    return "Genesis Chapter 01 Verse 01"


@pytest.fixture
def cal_word() -> PeshittaWord:
    """Return a mocked CAL word object not loaded from datasets."""
    return PeshittaWord(")lh", annots="noun sg. emphatic", origin="CAL")


@pytest.fixture
def etcbc_word() -> PeshittaWord:
    """Return a mocked ETCBC word object not loaded from datasets."""
    return PeshittaWord('BN"JH^', "ܒܢ̈ܝܗ̇", origin="ETCBC")


@pytest.fixture
def etcbc_chr_verse() -> Verse:
    """Return a Verse object for Peshitta 1 Chronicles 01:33."""
    return Verse(
        "Chronicles_1",
        "1 Chronicles Chapter 01 Verse 33",
        [
            'WB"NJ',
            "MDJN",
            "<P>",
            "W><PR",
            "WXNWK",
            "W>BJD<",
            "W>LR<>",
            "HLJN",
            "KLHWN",
            'BN"JH^',
            "DQNVWR>",
        ],
        syriac_words=[
            "ܘܒ̈ܢܝ",
            "ܡܕܝܢ",
            "ܥܦܐ",
            "ܘܐܥܦܪ",
            "ܘܚܢܘܟ",
            "ܘܐܒܝܕܥ",
            "ܘܐܠܪܥܐ",
            "ܗܠܝܢ",
            "ܟܠܗܘܢ",
            "ܒܢ̈ܝܗ̇",
            "ܕܩܢܛܘܪܐ",
        ],
        origin="ETCBC",
    )


# Code adapted from https://stackoverflow.com/a/21590140
# (accessed: 10 April 2025)
@pytest.fixture
def words(etcbc_word: PeshittaWord) -> object:
    """A factory fixture to replicate the same word multiple times.

    Returns:
        a factory class to make a list of the same ETCBC style PeshittaWord.
    """

    class ETCBCWordFactory:
        @staticmethod
        def copy(num: int) -> list[PeshittaWord]:
            result = []
            for i in range(num):
                result.append(etcbc_word)
            return result

    return ETCBCWordFactory()


@pytest.fixture
def etcbc_verse(genesis_1_1: str) -> Verse:
    """Returns an ETCBC style Verse for Peshitta Genesis 01:01."""
    return Verse(
        book_name="Genesis",
        verse_ref=genesis_1_1,
        translit_words=["BRCJT", "BR>", ">LH>", "JT", "CMJ>", "WJT", ">R<>"],
        syriac_words=["ܒܪܫܝܬ", "ܒܪܐ", "ܐܠܗܐ", "ܝܬ", "ܫܡܝܐ", "ܘܝܬ", "ܐܪܥܐ"],
        origin="ETCBC",
    )


@pytest.fixture
def etcbc_words() -> list[str]:
    """Returns a list of ETCBC style PeshittaWord for Peshitta Genesis 01:01."""
    words = ["BRCJT", "BR>", ">LH>", "JT", "CMJ>", "WJT", ">R<>"]
    syriac = ["ܒܪܫܝܬ", "ܒܪܐ", "ܐܠܗܐ", "ܝܬ", "ܫܡܝܐ", "ܘܝܬ", "ܐܪܥܐ"]
    res = []
    for i in range(len(words)):
        res.append(PeshittaWord(translit=words[i], syriac=syriac[i]))
    return res


@pytest.fixture
def cal_verse(genesis_1_1: str) -> Verse:
    """Returns a CAL style Verse for Peshitta Genesis 01:01."""
    return Verse(
        book_name="Genesis",
        verse_ref=genesis_1_1,
        translit_words=[
            "br$yt",
            "br)",
            ")lh)",
            "yt",
            "$my)",
            "w_",
            "yt",
            ")r()",
        ],
        words_annotations=[
            "noun sg. abs. or construct",
            "verb G",
            "noun sg. emphatic",
            "p01",
            "noun pl. emphatic",
            "c",
            "p01",
            "noun sg. emphatic",
        ],
    )


@pytest.fixture
def cal_words() -> list[PeshittaWord]:
    """Returns a list of CAL style PeshittaWord for Peshitta Genesis 01:01."""
    words = ["br$yt", "br)", ")lh)", "yt", "$my)", "w_", "yt", ")r()"]
    annotations = [
        "noun sg. abs. or construct",
        "verb G",
        "noun sg. emphatic",
        "p01",
        "noun pl. emphatic",
        "c",
        "p01",
        "noun sg. emphatic",
    ]
    res = []
    for i in range(len(words)):
        res.append(PeshittaWord(translit=words[i], annots=annotations[i]))
    return res


@pytest.fixture
def cal_translits() -> list[str]:
    """Return the transliteration of Genesis 01:01 as list of str."""
    return ["br$yt", "br)", ")lh)", "yt", "$my)", "w_", "yt", ")r()"]


@pytest.fixture
def cal_annots() -> list[str]:
    """Return the word annotations of Genesis 01:01 as list of str."""
    return [
        "noun sg. abs. or construct",
        "verb G",
        "noun sg. emphatic",
        "p01",
        "noun pl. emphatic",
        "c",
        "p01",
        "noun sg. emphatic",
    ]


@pytest.fixture
def cal_syriacs() -> list[str]:
    """Return the syriac script of Genesis 01:01 as list of str."""
    return ["ܒܪܫܝܬ", "ܒܪܐ", "ܐܠܗܐ", "ܝܬ", "ܫܡܝܐ", "ܘ", "ܝܬ", "ܐܪܥܐ"]


@pytest.fixture
def full_data_verse(
    cal_translits: list[str],
    cal_annots: list[str],
    cal_syriacs: list[str],
    genesis_1_1: str,
) -> Verse:
    """Returns a verse obj populated with CAL data for Genesis 01:01."""
    return Verse(
        book_name="Genesis",
        verse_ref=genesis_1_1,
        translit_words=cal_translits,
        syriac_words=cal_syriacs,
        words_annotations=cal_annots,
        origin="CAL",
    )


@pytest.fixture
def full_data_words(
    cal_translits: list[str], cal_annots: list[str], cal_syriacs: list[str]
) -> list[PeshittaWord]:
    """Returns a list of CAL style PeshittaWord objs for Genesis 01:01."""
    result = []
    for i in range(len(cal_translits)):
        result.append(
            PeshittaWord(
                cal_translits[i], cal_syriacs[i], cal_annots[i], origin="CAL"
            )
        )
    return result


@pytest.fixture
def cal_romans_verse() -> Verse:
    """Returns a CAL style NT Romans verse."""
    return Verse(
        "Romans",
        "Romans Chapter 01 Verse 01",
        [
            "p.awlAws",
            "(ab_d.A)",
            "d",
            "ye$w_(",
            "m$yixA)",
            "qaryA)",
            "w",
            "a$lyixA)",
            "d",
            "e)t_p.re$",
            "l",
            "e)wang.elyiAwn",
            "d",
            "a)lAhA)",
        ],
        syriac_words=[
            "ܦܿܰܘܠܳܘܣ",
            "ܥܰܒܼܕܿܳܐ",
            "ܕ",
            "ܝܶܫܽܘܥ",
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
        ],
        words_annotations=[
            "PN Personal Name",
            "noun sg. emphatic",
            "p",
            "PN Personal Name",
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
        ],
        origin="CAL",
    )


@pytest.fixture
def etcbc_chr_verses(etcbc_chr_verse: Verse) -> list[Verse]:
    """Returns a list of three ETCBC style Verses from 1 Chronicles."""
    res = [etcbc_chr_verse]
    res.append(
        Verse(
            "Chronicles_1",
            "1 Chronicles Chapter 01 Verse 34",
            ["W>WLD", ">BRHM", "L>JSXQ", 'BN"WHJ', "D>JSXQ", "<SW", "W>JSRJL"],
            ["ܘܐܘܠܕ", "ܐܒܪܗܡ", "ܠܐܝܣܚܩ", "ܒܢ̈ܘܗܝ", "ܕܐܝܣܚܩ", "ܥܣܘ", "ܘܐܝܣܪܝܠ"],
            origin="ETCBC",
        )
    )
    res.append(
        Verse(
            "Chronicles_1",
            "1 Chronicles Chapter 01 Verse 35",
            ['BN"WHJ', "D<SW", ">LJPZ", "WR<W>JL", "WJ<WC", "WJ<LJM", "WQWRX"],
            ["ܒܢ̈ܘܗܝ", "ܕܥܣܘ", "ܐܠܝܦܙ", "ܘܪܥܘܐܝܠ", "ܘܝܥܘܫ", "ܘܝܥܠܝܡ", "ܘܩܘܪܚ"],
            origin="ETCBC",
        )
    )
    return res


@pytest.fixture
def etcbc_acts_verse() -> Verse:
    """Returns an ETCBC style Verse containing Acts 01:06."""
    return Verse(
        "Acts",
        "Acts Chapter 01 Verse 06",
        translit_words=[
            "HNWN",
            "DJN",
            "KD",
            "KNJCJN",
            "C>LWHJ",
            "W>MRJN",
            "LH",
            "MRN",
            ">N",
            "BHN>",
            "ZBN>",
            "MPN>",
            ">NT",
            "MLKWT>",
            "L>JSRJL",
        ],
        syriac_words=[
            "ܗܢܘܢ",
            "ܕܝܢ",
            "ܟܕ",
            "ܟܢܝܫܝܢ",
            "ܫܐܠܘܗܝ",
            "ܘܐܡܪܝܢ",
            "ܠܗ",
            "ܡܪܢ",
            "ܐܢ",
            "ܒܗܢܐ",
            "ܙܒܢܐ",
            "ܡܦܢܐ",
            "ܐܢܬ",
            "ܡܠܟܘܬܐ",
            "ܠܐܝܣܪܝܠ",
        ],
    )


@pytest.fixture
def etcbc_cor1_verse() -> Verse:
    """Returns an ETCBC style Verse containing 1 Corinthians 02:07."""
    return Verse(
        "1_Corinthians",
        "1_Corinthians Chapter 02 Verse 07",
        [
            ">L>",
            "MMLLJNN",
            "XKMT>",
            "D>LH>",
            "B>RZ",
            "HJ",
            "DMKSJ>",
            "HWT",
            "WQDM",
            "HW>",
            "PRCH",
            ">LH>",
            "MN",
            "QDM",
            "<LM>",
            "LCWBX>",
            "DJLN",
        ],
        [
            "ܐܠܐ",
            "ܡܡܠܠܝܢܢ",
            "ܚܟܡܬܐ",
            "ܕܐܠܗܐ",
            "ܒܐܪܙ",
            "ܗܝ",
            "ܕܡܟܣܝܐ",
            "ܗܘܬ",
            "ܘܩܕܡ",
            "ܗܘܐ",
            "ܦܪܫܗ",
            "ܐܠܗܐ",
            "ܡܢ",
            "ܩܕܡ",
            "ܥܠܡܐ",
            "ܠܫܘܒܚܐ",
            "ܕܝܠܢ",
        ],
    )


@pytest.fixture
def cal_one_word_verse(cal_word: PeshittaWord, genesis_1_1: str) -> Verse:
    """Returns a CAL style Verse containing only the word God.

    The reference for this verse is pointed at Genesis 01:01.
    """
    return Verse(
        "Genesis",
        genesis_1_1,
        translit_words=[cal_word.translit],
        words_annotations=[cal_word.annots],
        origin="CAL",
    )


# Predictions & ProbaPredictions
@pytest.fixture
def proba_preds(
    cal_verse: Verse, etcbc_verse: Verse, full_data_verse: Verse
) -> ProbaPredictions:
    """Return a ProbaPredictions instance with verses from Genesis."""
    return ProbaPredictions(
        [cal_verse, etcbc_verse, full_data_verse],
        [0, 0, 1],
        [[0.8, 0.2], [0.7, 0.3], [0.4, 0.6]],
        [0, 0, 0],
    )


@pytest.fixture
def proba_preds_no_correct(
    etcbc_verse: Verse, etcbc_acts_verse: Verse
) -> ProbaPredictions:
    """Return a ProbaPredictions instance with no correct labels."""
    return ProbaPredictions(
        [etcbc_verse, etcbc_acts_verse], [0, 1], [[0.7, 0.3], [0.4, 0.6]]
    )


@pytest.fixture
def proba_preds_mislab(
    etcbc_chr_verses: list[Verse],
    etcbc_acts_verse: Verse,
    etcbc_cor1_verse: Verse,
) -> ProbaPredictions:
    """Return a ProbaPredictions instance with several mislabels."""
    verses = etcbc_chr_verses
    verses.extend([etcbc_acts_verse, etcbc_cor1_verse])
    return ProbaPredictions(
        verses,
        [1, 1, 0, 0, 0],
        [[0.11, 0.89], [0.33, 0.67], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
        [0, 0, 0, 1, 1],
    )


@pytest.fixture
def proba_multi_preds(
    cal_verse: Verse, etcbc_chr_verse: Verse
) -> ProbaPredictions:
    """Return a ProbaPredictions instance with verses of Gen and 1 Chr."""
    return ProbaPredictions(
        [cal_verse, etcbc_chr_verse],
        [0, 1],
        [[0.8, 0.2], [3.2e-10, 0.9]],
        [0, 0],
    )


@pytest.fixture
def proba_pred_zero(
    etcbc_verse: Verse,
    etcbc_acts_verse: Verse,
    etcbc_cor1_verse: Verse,
) -> ProbaPredictions:
    """Mock ProbaPredictions object containing zero proba for a verse."""
    return ProbaPredictions(
        [etcbc_verse, etcbc_acts_verse, etcbc_cor1_verse],
        [0, 1, 0],
        [[0.7, 0.3], [0.4, 0.6], [1.0, 0.0]],
        [0, 1, 1],
    )


# Mislabels
@pytest.fixture
def misls(etcbc_chr_verses: list[Verse]) -> Mislabels:
    """Return a Mislabels instance with multiple verses."""
    return Mislabels(
        [1, 1, 1],
        [0, 0, 0],
        etcbc_chr_verses,
        [[1.8e-10, 0.9], [0.667, 0.333], [0.401, 0.60]],
    )


@pytest.fixture
def mislabels_both(
    etcbc_verse: Verse,
    etcbc_cor1_verse: Verse,
) -> Mislabels:
    """Mock Mislabels with OT and NT verses."""
    return Mislabels(
        [1, 0],
        [0, 1],
        [etcbc_verse, etcbc_cor1_verse],
        [[0.3, 0.7], [0.6, 0.4]],
    )


# DataSplit
@pytest.fixture
def cal_ds(
    cal_verse: Verse, cal_one_word_verse: Verse, cal_romans_verse: Verse
) -> DataSplit:
    """Return a DataSplit object with CAL data."""
    return DataSplit([cal_verse, cal_one_word_verse], [cal_romans_verse])


@pytest.fixture
def etcbc_ds(
    etcbc_chr_verses: list[Verse],
    etcbc_acts_verse: Verse,
    etcbc_cor1_verse: Verse,
) -> DataSplit:
    """Return a DataSplit object with ETCBC data."""
    return DataSplit(etcbc_chr_verses, [etcbc_acts_verse, etcbc_cor1_verse])


@pytest.fixture
def loaded_etcbc(
    etcbc_chr_verses: list[Verse],
    etcbc_cor1_verse: Verse,
    etcbc_verse: Verse,
    etcbc_acts_verse: Verse,
    full_data_verse: Verse,
) -> LoadedDataset:
    """Return a LoadedDataset object with ETCBC data."""
    return LoadedDataset(
        etcbc_chr_verses,
        [etcbc_cor1_verse],
        [etcbc_verse],
        [etcbc_acts_verse],
        [full_data_verse],
    )


# Classifier
@pytest.fixture
def lr_classifier() -> Classifier:
    """Mock LinearRegression classifier."""
    # ArgumentType can be ignored here since LinearRegression inherits from the
    # BaseEstimator class
    return Classifier(LinearRegression())


@pytest.fixture
def mnb_classifier() -> BoWEstimator:
    """Mock MultinomialNB classifier."""
    return BoWEstimator(MultinomialNB(), identity, n=3)


@pytest.fixture
def mock_bow_clf():
    """Mock BoWEstimator object."""
    mock_clf = MagicMock(spec=BoWEstimator)
    mock_clf.fit.return_value = None
    mock_clf.predict_proba.return_value = [[0.8, 0.2], [0.6, 0.4]]
    return mock_clf


@pytest.fixture
def mock_formatter(
    samples: list[Verse] | NDArray[Verse],
    probas: list[list[float]] | NDArray[np.float64],
    correct_labels: list[int] | NDArray[np.int64] | None = None,
) -> str:
    """Mock formatter implementing FileFormatterProto."""
    if correct_labels is not None:
        return "A,B,C,D,E"
    # else
    return "A,B,C,D"


@pytest.fixture
def mock_predict_proba(proba_preds: ProbaPredictions) -> MagicMock:
    return MagicMock(return_value=proba_preds)


@pytest.fixture
def mock_eval_and_save():
    """Mock eval_and_save function."""
    return MagicMock(return_value=None)


# Threshold-related Results
@pytest.fixture
def thresh_stats_1() -> ThresholdStats:
    return ThresholdStats(
        threshold=0.6,
        accuracy=0.85,
        precision=[0.7, 0.8],
        recall=[0.6, 0.7],
        f_beta=[0.65, 0.75],
    )


@pytest.fixture
def thresh_stats_2() -> ThresholdStats:
    return ThresholdStats(
        threshold=0.5,
        accuracy=0.9,
        precision=[0.8, 0.9],
        recall=[0.1, 0.8],
        f_beta=[0.75, 0.85],
    )


@pytest.fixture
def result_stats_1(
    thresh_stats_1: ThresholdStats, thresh_stats_2: ThresholdStats
) -> ResultStats:
    stats = ResultStats(
        supports=[50, 30],
        log_loss=[0.2, 0.3],
        roc_auc=[0.9, 0.85],
    )
    thresh_stats = [thresh_stats_1, thresh_stats_2]
    stats.add_thresh_stats(thresh_stats)
    return stats


# SavefileName
@pytest.fixture
def cal_base_savefile() -> SavefileName:
    return SavefileName(origin="CAL", classifier_alias="mnb", file_ext="csv")


@pytest.fixture
def etcbc_base_savefile() -> SavefileName:
    return SavefileName(origin="ETCBC", classifier_alias="svc", file_ext="json")


# BookData
@pytest.fixture
def cal_book_data_gen() -> BookData:
    """Mock BookData dictionary."""
    return {
        "book_title": "Genesis",
        "verse_refs": ["Genesis Chapter 01 Verse 01"],
        "lemmatised_verses": [
            ["br$yt", "br)", ")lh)", "yt", "$my)", "w_", "yt", ")r()"]
        ],
        "lemma_annotations": [
            [
                "noun sg. abs. or construct",
                "verb G",
                "noun sg. emphatic",
                "p01",
                "noun pl. emphatic",
                "c",
                "p01",
                "noun sg. emphatic",
            ],
        ],
    }


@pytest.fixture
def cal_book_data_est() -> BookData:
    return {
        "book_title": "Esther",
        "verse_refs": [
            "Esther Chapter 00 Verse 00",
            "Esther Chapter 01 Verse 01",
            "Esther Chapter 01 Verse 02",
        ],
        "lemmatised_verses": [
            ["ktb", "d_", ")styr"],
            [
                "w_",
                "hwy",
                "b_",
                "ywm",
                "d_",
                ")x$yr$",
                "hw",
                "br",
                "d_",
                ")x$yr$",
                "d_",
                "mlk",
                "mn",
                "hwd",
                "w_",
                "(dm)",
                "l_",
                "kw$",
                "(l",
                "m))",
                "w_",
                "(sryn",
                "mdynh",
            ],
            [
                "b_",
                "ywm",
                "hnwn",
                "kd",
                "ytb",
                "hwy",
                "mlk",
                ")x$yr$",
                "(l",
                "kwrsy",
                "d_",
                "mlkw",
                "d_",
                "b_",
                "$w$n",
                "byrh",
            ],
        ],
        "lemma_annotations": [
            ["noun sg. emphatic", "p = d_ p --> dy p", "PN Personal name"],
            [
                "c",
                "verb G",
                "p02",
                "noun pl. emphatic",
                "p = d_ p --> dy p",
                "PN Personal name",
                "P01",
                "noun sg. emphatic",
                "p = d_ p --> dy p",
                "PN Personal name",
                "c = d_ c --> dy c",
                "verb C",
                "p01",
                "GN Geographic name",
                "c",
                "c = (dm) c --> (dm) p",
                "p03",
                "GN Geographic name",
                "p01",
                "n01 = m)) b --> m)h b",
                "c",
                "n01",
                "noun pl. absolute",
            ],
            [
                "p02",
                "noun pl. emphatic",
                "P01",
                "c",
                "verb G",
                "verb G",
                "noun sg. emphatic",
                "PN Personal name",
                "p01",
                "noun sg. emphatic",
                "p = d_ p --> dy p",
                "noun sg. emphatic",
                "c = d_ c --> dy c",
                "p02",
                "GN Geographic name",
                "noun sg. emphatic",
            ],
        ],
    }


@pytest.fixture
def cal_books_data(cal_book_data_gen, cal_book_data_est) -> list[BookData]:
    return [cal_book_data_gen, cal_book_data_est]
