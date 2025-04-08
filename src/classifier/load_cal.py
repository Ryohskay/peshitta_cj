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

from pathlib import Path
from typing import NamedTuple

import pandas as pd

from classifier import book_data
from classifier.dataset_skeleton import LoadedDataset
from classifier.result_utils import Verse


def normalise_title(title: str) -> str:
    """Normalise the title of biblical books according to ETCBC.

    Returns:
        normalised name of the biblical book.

    Examples:
        >>> classifier.load_cal.normalise_title("1_Samuel")
        >>> "Samuel_1"
    """
    if "_" in title:
        # 1_Samuel to Samuel_1
        return "_".join(title.split("_")[::-1])
    if "Nehemiah":
        return "Nehemia"
    return title


def load_df_json(data_dir: str | Path) -> pd.DataFrame | None:
    """Get CAL scraper results in json format and load as pd.DataFrame.

    Returns:
        pd.DataFrame object with the CAL scraper results.
    """
    total_df = None
    for fpath in Path(data_dir).iterdir():
        # Load the json data
        df = pd.read_json(fpath)
        # Append to total_df
        if total_df is None:
            total_df = df
        else:
            total_df = pd.concat([total_df, df], ignore_index=True)
    return total_df


def get_book(df: pd.DataFrame, title: str) -> pd.DataFrame:
    """Filter the DataFrame by book_title column.

    Returns:
        A column from the ``df`` only containing the "book_title" column.
    """
    return df.loc[df["book_title"] == title]


def verse_in_chapters(verse: NamedTuple, chapters: list[int]) -> bool:
    """Check if the verse is in the list of provided chapters.

    Returns:
        True if the verse's chapter is in the ``chapters`` list.

    Raises:
        AttributeError: if the provided argument ``verse`` is not a NamedTuple
            expected by the code.
    """
    try:
        return int(verse.verse_refs.split(" ")[2]) in chapters  # type: ignore[reportAttributeAccessIssue]
    except AttributeError as ae:
        msg = ("Attribute or method was not found. Please check that: "
               + f"verse (type {type(verse)}) "
               + ".verse_refs (type {type(verse_refs)}) has a method `split()`")
        raise AttributeError(msg) from ae


def get_book_verses(
    df: pd.DataFrame, target_books: dict, *, trim_none: bool = False
) -> list[Verse]:
    """Get all books in the dataframe and generate a list of tuples.

    Returns:
        list of :class:``Verse`` instances
    """
    books = None
    verse_box = []

    load_books = list(target_books.keys())  # books to load
    for bk in load_books:
        print(bk)
        book_df = get_book(df, bk)
        if books is None:
            # if this is the first iteration,
            # just load ``book_df``
            books = book_df
        else:
            # else, concatenate the loaded ``book_df with`` exising ``books``
            books = pd.concat([books, book_df], ignore_index=True)

        # extract specific columns from the loaded dataframe
        selected_cols = books[
            ["verse_refs", "lemmatised_verses", "lemma_annotations"]
        ]
        # get data in each row as a named tuple
        for verse in selected_cols.itertuples():
            # if verse is in target chapters
            if verse_in_chapters(verse, target_books[bk]):  # type: ignore[reportAttributeAccessIssue]
                refs = verse.verse_refs  # type: ignore[reportAttributeAccessIssue]
                lemmata = []  # list[lemma]
                lemma_annots = []  # list[(lemma, annotations)]
                for i in range(len(verse.lemmatised_verses)):  # type: ignore[reportAttributeAccessIssue]
                    lemma = verse.lemmatised_verses[i]  # type: ignore[reportAttributeAccessIssue]
                    annot = verse.lemma_annotations[i]  # type: ignore[reportAttributeAccessIssue]
                    annot_r = annot if annot is not None else ""
                    # Handle lemma
                    if (trim_none and lemma is not None) or not trim_none:
                        lemmata.append(lemma)
                        lemma_annots.append(annot_r)
                    else:
                        # in some cases, the scraper records the lemma as None.
                        # we don't need them so skip appending lemma and annots
                        continue
                verse_box.append(
                        Verse(bk, refs,
                                    lemmata, words_annotations=lemma_annots,
                                origin="CAL"
                              )
                        )
    # print("Loaded CAL")
    # print(f"{type(verse_box[0])}")
    # print(f"{verse_box[0]}")
    return verse_box


def load_cal_dataset(
        src_dir: str = "./"
    ) -> LoadedDataset:
    """Load CAL dataset from json files into a dictionary.

    .. note::
        To change the books and chapters from which the data is loaded, update
        :mod:`classifier.book_data`.

    Args:
        src_dir: string containing path to the ``src`` directory, or where
            ``scraper/cal_results/`` is located.

    Returns:
        a :class:`classifier.dataset_skeleton.LoadedDataset` instance.

    Raises:
        RuntimeError: if the directory for json files exists but could not
            load data from there.
    """
    # load the CSV data files
    proj_root = Path(src_dir)
    # Make sure you don't add slash at the beginning of the second file path
    target_path = proj_root / "scraper/cal_results/"
    print(f"Loading dataset(s) from: {target_path}")

    # Load the data into Pandas' DataFrame for easier control
    df = load_df_json(target_path)
    if df is None:
        msg = f"Failed to load data from {target_path}"
        raise RuntimeError(msg)

    # get the training data
    print("OT_train")
    ot_train_verses = get_book_verses(
        df, book_data.ot_train_books, trim_none=True
    )
    print("NT_train")
    nt_train_verses = get_book_verses(
        df, book_data.nt_train_books, trim_none=True
    )

    # get the test data
    ot_test_verses = get_book_verses(
        df, book_data.ot_test_books, trim_none=True
    )
    nt_test_verses = get_book_verses(
        df, book_data.nt_test_books, trim_none=True
    )

    # get the production data
    prod_verses = get_book_verses(
                    df, book_data.ot_prod_books, trim_none=True
                    )
    return LoadedDataset(ot_train_verses, nt_train_verses,
                                ot_test_verses, nt_test_verses, prod_verses)



if __name__ == "__main__":
    d = load_cal_dataset("./")
    print(d.test.get_samples())
    print(d.test.get_labels())
