"""Test the code to load ETCBC data through the textfabric library."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from tf import app

from src.classifier.dataset_skeleton import LoadedDataset
from src.classifier.result_utils import Verse
from src.classifier.textfabric_utils import get_verses, load_etcbc_dataset


@pytest.fixture
def targets():
    return {
        "Genesis": [1],  # Genesis chapter 1
        "Exodus": [1, 2],  # Exodus chapter 1, 2
    }


def ladder_down(key: int, otype: str):  # noqa: ARG001
    map_ids = {
        # book -> chapter
        1: [101, 102],  # Gen ch 1,2
        2: [103],  # Ex ch 1
        # chapter -> verse
        101: [201, 202],  # Gen ch 1 v 1,2
        102: [203],  # Gen ch 2 v 1
        103: [204],  # Ex ch 1 v 1
        # verse -> word
        201: [301, 302, 303],  # Gen ch 1 v 1
        202: [304],  # Gen ch 1 v 2
        203: [305, 306],  # Gen ch 2 v 1
        204: [307],  # Ex ch 1 v 1
    }
    return map_ids[key]


def _mock_f_internal(level: str) -> Callable[[int], str]:
    def res(val: int) -> str:
        if level == "chapter":
            return {
                # chapter ids
                101: "1",  # Gen ch 1
                102: "2",  # Gen ch 2
                103: "1",  # Ex ch 1
                # verse ids
                201: "1",
                202: "1",
                203: "2",
                204: "1",
                # word ids
                301: "1",
                302: "1",
                303: "1",
                304: "1",
                305: "2",
                306: "2",
                307: "1",
            }[val]
        if level == "verse":
            return {
                # verse ids
                201: "1",
                202: "2",  # Gen ch 1 v 1, 2
                203: "1",  # Gen ch 2 v 1
                204: "1",  # Ex ch 1 v 1
                # word ids
                301: "1",
                302: "1",
                303: "1",
                304: "2",
                305: "1",
                306: "1",
                307: "1",
            }[val]
        if level == "word_etcbc":
            return {
                301: "BRCJT",
                302: "BR>",
                303: ">LH>",
                304: ">R<>",
                305: "WCL#MW",
                306: "CMJ>",
                307: "WHLJN",
            }[val]
        # else
        return {
            301: "ܒܪܫܝܬ",
            302: "ܒܪܐ",
            303: "ܐܠܗܐ",
            304: "ܐܪܥܐ",
            305: "ܘܫܠ̣ܡܘ",
            306: "ܫܡܝܐ",
            307: "ܘܗܠܝܢ",
        }[val]

    return res


class MockTextFabricAPIHandle:
    def __init__(self) -> None:
        self.api = MagicMock()
        self.api.Fs.return_value = {1: "Genesis", 2: "Exodus"}
        self.api.L.d.side_effect = ladder_down
        self.api.F.chapter.v = _mock_f_internal("chapter")
        self.api.F.verse.v = _mock_f_internal("verse")
        self.api.F.word_etcbc.v = _mock_f_internal("word_etcbc")
        self.api.F.word.v = _mock_f_internal("word")


def _mock_use(
    *args,  # noqa: ANN002 ARG001
    **kwargs,  # noqa: ANN003, ARG001
) -> MockTextFabricAPIHandle:
    return MockTextFabricAPIHandle()


def test_get_verses(monkeypatch, targets: dict):  # noqa: ANN001
    with monkeypatch.context() as m:
        m.setattr(app, "use", _mock_use)

        target_books = targets
        verses = get_verses(target_books)

        assert len(verses) == 3
        assert isinstance(verses[0], Verse)
        assert verses[0].book == "Genesis"
        assert verses[0].reference == "Genesis Chapter 01 Verse 01"
        assert verses[0].get_translit_words() == ["BRCJT", "BR>", ">LH>"]
        assert verses[0].get_syriac_words() == ["ܒܪܫܝܬ", "ܒܪܐ", "ܐܠܗܐ"]
        assert verses[1].book == "Genesis"
        assert verses[1].reference == "Genesis Chapter 01 Verse 02"
        assert verses[2].book == "Exodus"
        assert verses[2].reference == "Exodus Chapter 01 Verse 01"


def test_load_etcbc_dataset(monkeypatch):  # noqa: ANN001
    with monkeypatch.context() as m:
        m.setattr(app, "use", _mock_use)

        dataset = load_etcbc_dataset(
            ot_train={"Genesis": [1]},
            nt_train={"Genesis": [2]},
            ot_test={"Genesis": [1]},
            nt_test={"Genesis": [2]},
            ot_prod={"Exodus": [1]},
        )

        assert isinstance(dataset, LoadedDataset)
        # check training set
        assert len(dataset.train.get_samples()) == 3
        assert len(dataset.train.get_samples(0)) == 2
        assert len(dataset.train.get_samples(1)) == 1
        assert dataset.train.get_samples()[0].book == "Genesis"
        assert (
            dataset.train.get_samples()[0].reference
            == "Genesis Chapter 01 Verse 01"
        )
        assert dataset.train.get_samples()[1].book == "Genesis"
        assert (
            dataset.train.get_samples()[1].reference
            == "Genesis Chapter 01 Verse 02"
        )
        assert dataset.train.get_samples()[2].book == "Genesis"
        assert (
            dataset.train.get_samples()[2].reference
            == "Genesis Chapter 02 Verse 01"
        )
        # check test set
        assert len(dataset.test.get_samples()) == 3
        assert len(dataset.test.get_samples(0)) == 2
        assert len(dataset.test.get_samples(1)) == 1
        assert dataset.test.get_samples()[0].book == "Genesis"
        assert (
            dataset.test.get_samples()[0].reference
            == "Genesis Chapter 01 Verse 01"
        )
        assert dataset.test.get_samples()[1].book == "Genesis"
        assert (
            dataset.test.get_samples()[1].reference
            == "Genesis Chapter 01 Verse 02"
        )
        assert dataset.test.get_samples()[2].book == "Genesis"
        assert (
            dataset.test.get_samples()[2].reference
            == "Genesis Chapter 02 Verse 01"
        )
        # check production set
        assert len(dataset.production) == 1
        assert dataset.production[0].book == "Exodus"
        assert dataset.production[0].reference == "Exodus Chapter 01 Verse 01"
