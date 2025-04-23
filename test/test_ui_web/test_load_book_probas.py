from pathlib import Path

import pytest

from src.classifier.fname_utils import FnameExtraOpts, SavefileName
from src.ui_web.load_book_probas import BookProbas, load_book_probas


def test_book_probas_init():
    """Test the initialization of the BookProbas class."""
    book_probas = BookProbas(origin="CAL")
    assert book_probas.origin == "CAL"
    assert book_probas.book_proba_dict == {}


def test_add_book_proba():
    """Test the add_book_proba method of the BookProbas class."""
    book_probas = BookProbas(origin="CAL")
    book_probas.add_book_proba("Genesis", [0.8, 0.2])
    assert book_probas.book_proba_dict == {"Genesis": [0.8, 0.2]}

    # Add another book
    book_probas.add_book_proba("Exodus", [0.6, 0.4])
    assert book_probas.book_proba_dict == {
        "Genesis": [0.8, 0.2],
        "Exodus": [0.6, 0.4],
    }


def test_load_book_probas(mock_save_fname: SavefileName):
    """Test the load_book_probas function."""
    assets_dir = Path("./assets/classifier_results")

    # Call the function
    book_probas = load_book_probas(mock_save_fname, load_dir=assets_dir)

    # Assertions
    assert isinstance(book_probas, BookProbas)
    assert book_probas.origin == "CAL"
    assert book_probas.book_proba_dict == {
        "Deuteronomy": [1.0, 2.628171465924225e-86],
        "Acts": [1.6434677447268384e-227, 1.0],
    }
