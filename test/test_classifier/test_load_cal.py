from src.classifier.dataset_skeleton import LoadedDataset
from src.classifier.load_cal import (
    BookData,
    extract_verse,
    get_book_verses,
    load_cal_dataset,
    load_json,
    verse_in_chapters,
)
from src.classifier.result_utils import Verse
from test.conftest import cal_verse


def test_load_json(cal_book_data_est: BookData):
    """Test the load_json function."""
    result = load_json("./assets/scraper/cal_results/")

    assert len(result) == 1
    assert result[0]["book_title"] == "Esther"
    assert len(result[0]["lemmatised_verses"]) == 3
    assert len(result[0]["verse_refs"]) == 3
    assert len(result[0]["lemma_annotations"]) == 3
    assert result[0]["book_title"] == cal_book_data_est["book_title"]
    assert (
        result[0]["lemmatised_verses"] == cal_book_data_est["lemmatised_verses"]
    )
    assert result[0]["verse_refs"] == cal_book_data_est["verse_refs"]
    assert (
        result[0]["lemma_annotations"] == cal_book_data_est["lemma_annotations"]
    )


def test_verse_in_chapters():
    """Test the verse_in_chapters function."""
    assert verse_in_chapters("Genesis Chapter 01 Verse 01", [1])
    assert not verse_in_chapters("Genesis Chapter 02 Verse 01", [1])
    assert not verse_in_chapters("1_Chronicles Chapter 05 Verse 01", [1])


def test_extract_verse(
    cal_book_data_gen: BookData, cal_translits: list[str], cal_annots: list[str]
):
    """Test the extract_verse function."""
    verse = extract_verse(cal_book_data_gen, 0)
    assert isinstance(verse, Verse)
    assert verse.book == "Genesis"
    assert verse.reference == "Genesis Chapter 01 Verse 01"
    assert verse.get_translit_words() == cal_translits
    assert verse.get_annotations() == cal_annots


def test_get_book_verses(cal_books_data: list):
    """Test the get_book_verses function."""
    target_books = {"Genesis": [1]}
    verses = get_book_verses(cal_books_data, target_books)

    assert len(verses) == 1
    assert isinstance(verses[0], Verse)
    assert verses[0].book == "Genesis"
    assert verses[0].reference == "Genesis Chapter 01 Verse 01"


def test_load_cal_dataset(
        cal_books_data: list[BookData],
        cal_verse: Verse
        ):
    """Test the load_cal_dataset function."""
    dataset = load_cal_dataset(
        src_dir="./assets/",
        ot_train={"Esther": [1]},
        nt_train={"Esther": [1]},
        ot_test={"Esther": [0]},
        nt_test={"Esther": [6]},
    )

    assert isinstance(dataset, LoadedDataset)
    assert len(dataset.train.get_samples()) == 4
    assert len(dataset.test.get_samples()) == 1
    assert dataset.test.get_samples()[0].book == "Esther"
    assert (
        dataset.train.get_samples()[0].reference == "Esther Chapter 01 Verse 01"
    )
