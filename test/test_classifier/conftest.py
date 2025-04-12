"""pytest.fixture objects to reuse among tests"""
import pytest

from classifier.result_utils import PeshittaWord, Verse


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
    return Verse("Chronicles_1",
                  "1 Chronicles Chapter 01 Verse 33",
                  ['WB"NJ', "MDJN", "<P>", "W><PR", "WXNWK", "W>BJD<", "W>LR<>", "HLJN", "KLHWN", 'BN"JH^', "DQNVWR>"],
                  syriac_words=["ܘܒ̈ܢܝ", "ܡܕܝܢ", "ܥܦܐ", "ܘܐܥܦܪ", "ܘܚܢܘܟ", "ܘܐܒܝܕܥ", "ܘܐܠܪܥܐ", "ܗܠܝܢ", "ܟܠܗܘܢ", "ܒܢ̈ܝܗ̇", "ܕܩܢܛܘܪܐ"],
                  origin="ETCBC"
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
    return Verse(book_name="Genesis",
          verse_ref=genesis_1_1,
          translit_words=["BRCJT", "BR>", ">LH>", "JT", "CMJ>", "WJT", ">R<>"],
          syriac_words=["ܒܪܫܝܬ", "ܒܪܐ", "ܐܠܗܐ", "ܝܬ", "ܫܡܝܐ", "ܘܝܬ",
                        "ܐܪܥܐ"],
          origin="ETCBC"
          )


@pytest.fixture
def etcbc_words() -> list[str]:
    """Returns a list of ETCBC style PeshittaWord for Peshitta Genesis 01:01."""
    words = ["BRCJT", "BR>", ">LH>", "JT", "CMJ>", "WJT", ">R<>"]
    syriac = ["ܒܪܫܝܬ", "ܒܪܐ", "ܐܠܗܐ", "ܝܬ", "ܫܡܝܐ", "ܘܝܬ",
                        "ܐܪܥܐ"]
    res = []
    for i in range(len(words)):
        res.append(PeshittaWord(translit=words[i], syriac=syriac[i]))
    return res


@pytest.fixture
def cal_verse(genesis_1_1: str) -> Verse:
    """Returns a CAL style Verse for Peshitta Genesis 01:01."""
    return Verse(book_name="Genesis",
        verse_ref=genesis_1_1,
        translit_words=["br$yt", "br)", ")lh)", "yt", "$my)", "w_",
                      "yt", ")r()"],
        words_annotations=["noun sg. abs. or construct", "verb G",
                         "noun sg. emphatic", "p01", "noun pl. emphatic",
                         "c", "p01", "noun sg. emphatic"
                         ]
    )


@pytest.fixture
def cal_words() -> list[PeshittaWord]:
    """Returns a list of CAL style PeshittaWord for Peshitta Genesis 01:01."""
    words = ["br$yt", "br)", ")lh)", "yt", "$my)", "w_",
                      "yt", ")r()"]
    annotations = ["noun sg. abs. or construct", "verb G",
                     "noun sg. emphatic", "p01", "noun pl. emphatic",
                     "c", "p01", "noun sg. emphatic"
                     ]
    res = []
    for i in range(len(words)):
        res.append(PeshittaWord(translit=words[i], annots=annotations[i]))
    return res


@pytest.fixture
def cal_translits() -> list[str]:
    return (["br$yt", "br)", ")lh)", "yt", "$my)", "w_",
                              "yt", ")r()"])


@pytest.fixture
def cal_annots() -> list[str]:
    return ["noun sg. abs. or construct", "verb G",
                 "noun sg. emphatic", "p01", "noun pl. emphatic",
                 "c", "p01", "noun sg. emphatic"
                 ]


@pytest.fixture
def cal_syriacs() -> list[str]:
    return ["ܒܪܫܝܬ", "ܒܪܐ", "ܐܠܗܐ", "ܝܬ", "ܫܡܝܐ", "ܘ", "ܝܬ", "ܐܪܥܐ"]


@pytest.fixture
def full_data_verse(
        cal_translits: list[str],
        cal_annots: list[str],
        cal_syriacs: list[str],
        genesis_1_1: str,
        ) -> Verse:
    return Verse(book_name="Genesis",
                 verse_ref=genesis_1_1,
                 translit_words=cal_translits,
                 syriac_words=cal_syriacs,
                 words_annotations=cal_annots
               )


@pytest.fixture
def full_data_words(cal_translits: list[str],
                    cal_annots: list[str],
                    cal_syriacs: list[str]
                    ) -> list[PeshittaWord]:
    result = []
    for i in range(len(cal_translits)):
        result.append(PeshittaWord(
                cal_translits[i],
                cal_syriacs[i],
                cal_annots[i],
                origin="CAL"
            ))
    return result


@pytest.fixture
def cal_one_word_verse(
        cal_word: PeshittaWord,
        genesis_1_1: str
        ) -> Verse:
    return Verse("Genesis",
          genesis_1_1,
          translit_words=[cal_word.translit],
          words_annotations=[cal_word.annots],
          origin="CAL"
          )
