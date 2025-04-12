# BSD 2-Clause License

# Copyright (c) 2025, Ryosuke Nagata

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Load data from CAL scraper results."""

import json
from pathlib import Path
from typing import TypedDict

from classifier import book_data
from src.classifier.dataset_skeleton import LoadedDataset
from src.classifier.result_utils import Verse


class BookData(TypedDict):
    book_title: str
    verse_refs: list[str]
    lemmatised_verses: list[list[str]]
    lemma_annotations: list[list[str]]


def normalise_title(book_title: str) -> str:
    # Mapping of book titles where ETCBC and CAL diverge.
    etcbc_cal_map = {
                "1_Chronicles": "Chronicles_1",
                "1_Samuel": "Samuel_1",
                "2_Samuel": "Samuel_2",
                "1_Kings": "Kings_1",
                "2_Kings": "Kings_2",
                "Nehemiah": "Nehemia",
                "2_Chronicles": "Chronicles_2",
                # "Maccabees_1_A": list(range(1, 16 + 1)),
                "1_Maccabees": "Maccabees_1_B",
            }

    if book_title in etcbc_cal_map:
        return etcbc_cal_map[book_title]
    # else
    return book_title


def load_json(data_dir: str | Path) -> list[BookData]:
    """Get CAL scraper results in json format and load as dict.

    Returns:
        list of tuple containing a book title and a dict with the CAL scraper
        results for that book.

    Raises:
        RuntimeError: if the directory for json files exists but could not
            load data from there.

    Todo:
        validate json with jsonschema
    """
    total_data = []

    for file in Path(data_dir).iterdir():
        with file.open() as fp:
            json_obj = json.load(fp)
            total_data.append(json_obj)

    if len(total_data) < 1:  # if total_data is empty after the loop
        msg = f"Failed to load data from {data_dir}"
        raise RuntimeError(msg)

    return total_data


def verse_in_chapters(verse_ref: str, chapters: list[int]) -> bool:
    """Check if the verse is in the list of provided chapters.

    Returns:
        True if the verse's chapter is in the ``chapters`` list.
    """
    try:
        return int(verse_ref.split(" ")[2]) in chapters
    except IndexError:
        # index out of range can occur with things like "24:XX"
        return False


def extract_verse(
        book_dict: BookData, verse_idx: int,
        *, trim_none: bool = False
    ) -> Verse:
    refs = book_dict["verse_refs"][verse_idx]
    lemmata = []  # list[lemma]
    lemma_annots = []  # list[annotations]
    if not trim_none:
        lemmata = book_dict["lemmatised_verses"][verse_idx]
        lemma_annots = book_dict["lemma_annotations"][verse_idx]
    else:  # if trim_none option is specified
        for j in range(len(book_dict["lemmatised_verses"][verse_idx])):
            lemma = book_dict["lemmatised_verses"][verse_idx][j]
            annot = book_dict["lemma_annotations"][verse_idx][j]
            annot_r = annot if annot is not None else ""
            # Handle lemma
            if lemma is not None:
                lemmata.append(lemma)
                lemma_annots.append(annot_r)
            else:
                # in some cases, the scraper records the lemma as None.
                # we don't need them so skip appending lemma and annots
                continue

    return Verse(book_dict["book_title"],
          refs, lemmata,
          words_annotations=lemma_annots, origin="CAL")


def get_book_verses(
        jso_lis: list[BookData], target_books: dict[str, list[int]],
        *, trim_none: bool = False
    ) -> list[Verse]:
    """Get all books in the dict format and return a list of ``Verse``.

    This function checks if each verse in the listed books is target for
    extraction, and if so, calls :func:`src.classifier.load_cal.extract_verse`
    to get a :class:`src.classifier.result_utils.Verse` object.

    Returns:
        list of :class:`src.classifier.result_utils.Verse` instances
    """
    verse_box: list[Verse] = []
    for book in jso_lis:
        # one book
        book["book_title"] = normalise_title(book["book_title"])
        if book["book_title"] in target_books:
            # print(book["book_title"])
            for i in range(len(book["lemmatised_verses"])):
                # one verse
                if (verse_in_chapters(
                        book["verse_refs"][i],
                        target_books[book["book_title"]]
                    )):
                    verse_box.append(  # noqa: PERF401
                            extract_verse(book, i, trim_none=trim_none)
                            )
    return verse_box


def load_cal_dataset(
        src_dir: str = "./"
    ) -> LoadedDataset:
    """Load CAL dataset from json files into a dictionary.

    .. note::
        To change the books and chapters from which the data is loaded, update
        :mod:`src.classifier.book_data`.

    Args:
        src_dir: string containing path to the ``src`` directory, or where
            ``scraper/cal_results/`` is located.

    Returns:
        a :class:`src.classifier.dataset_skeleton.LoadedDataset` instance.
    """
    # load the CSV data files
    proj_root = Path(src_dir)
    # Make sure you don't add slash at the beginning of the second file path
    target_path = proj_root / "scraper/cal_results/"
    print(f"Loading dataset(s) from: {target_path.resolve()}")

    # Load the data into Pandas' DataFrame for easier control
    loaded_data = load_json(target_path)

    # get the training data
    ot_train_verses = get_book_verses(
        loaded_data, book_data.ot_train_books, trim_none=True
    )
    nt_train_verses = get_book_verses(
        loaded_data, book_data.nt_train_books, trim_none=True
    )

    # get the test data
    ot_test_verses = get_book_verses(
        loaded_data, book_data.ot_test_books, trim_none=True
    )
    nt_test_verses = get_book_verses(
        loaded_data, book_data.nt_test_books, trim_none=True
    )

    # get the production data
    prod_verses = get_book_verses(
                    loaded_data, book_data.ot_prod_books, trim_none=True
                    )
    return LoadedDataset(ot_train_verses, nt_train_verses,
                                ot_test_verses, nt_test_verses, prod_verses)


if __name__ == "__main__":
    d = load_cal_dataset("./")
    print(d.test.get_samples())
    print(d.test.get_labels())
