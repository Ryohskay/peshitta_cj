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
"""Load predictions from CSV files."""

import csv
import logging
from pathlib import Path
from typing import TypedDict

from src.classifier.fname_utils import SavefileName
from src.classifier.result_utils import Verse

logger = logging.getLogger(__name__)


class BookVerses(TypedDict):
    """A dict to represent the link between verses and a book."""

    book_name: str  # The name of the book.
    verses: list[Verse]  # A list of Verse instances.
    # list of lists containing the predicted probabilities for
    # each verse for each class.
    verse_probas: list[list[float]]


def load_preds(
    fname: SavefileName,
    load_dir: str | Path = Path("src/classifier/out/"),
) -> list[BookVerses]:
    """Load prediction results from a CSV file.

    Args:
        origin_name: name of the database or dataset(s) the data was originally
            extracted from.
        fname: name of the target CSV file. You should use the one with "all"
            in the name since such a file contains all predictions on a dataset.
        load_dir: str indicating path to the directory where the CSV file is
            located.
        threshold: threshold of predicted probability to classify a sample to be
            belonging to a particular class.

    Returns:
        list of :class:`src.ui_web.load_predictions.BookVerses` instances
        containing the verses and their prediction results from each book.
    """
    data_origin = fname.origin
    fname_p = Path(load_dir) / fname.get_fname()
    parsed_books = []
    books: list[BookVerses] = []

    print(f"Loading predictions from {fname_p}")
    with fname_p.open(newline="", encoding="utf-8") as csvfile:
        read_data = csv.reader(csvfile)
        first_row = True
        current_book = ""
        verses: list[Verse] = []
        probas: list[list[float]] = []
        for row in read_data:
            if first_row:
                # skip the header row
                first_row = False
                continue

            if (
                current_book
                and current_book != row[0]
                and current_book not in parsed_books
            ):
                # when we finish parsing a book
                parsed_books.append(current_book)
                books.append(
                    {
                        "book_name": current_book,
                        "verses": verses,
                        "verse_probas": probas,
                    }
                )
                # reset the lists
                verses = []
                probas = []

            if current_book != row[0]:
                # when we discover a new book
                current_book = row[0]
            log_msg = f"(Ref: {row[1]}) OT > {row[2]} NT > {row[3]} [{row[4]}]"
            logger.info(log_msg)
            probas.append([float(row[2]), float(row[3])])

            # initialise the Verse instance based on the data origin
            if data_origin == "CAL":
                verses.append(
                    Verse(row[0], row[1], row[4].split(" "), origin=data_origin)
                )
            elif data_origin == "ETCBC":
                verses.append(
                    Verse(
                        row[0],
                        row[1],
                        row[4].split(" "),
                        syriac_words=row[5].split(" "),
                        origin=data_origin,
                    )
                )
        # when we finish parsing the last book
        books.append(
            {
                "book_name": current_book,
                "verses": verses,
                "verse_probas": probas,
            }
        )
    return books


if __name__ == "__main__":
    outdir = "src/classifier/out/"
    # threshold
    threshold = 0.5

    # CAL
    cal_fname = SavefileName("CAL", "mnb", ".csv")
    cal_fname.set_ngram_opts(n=3)
    cal_fname.mark_special_file(is_prod=True)
    load_preds(cal_fname, outdir)

    # ETCBC
    etcbc_fname = SavefileName("ETCBC", "mnb", ".csv")
    etcbc_fname.set_ngram_opts(n=3)
    etcbc_fname.mark_special_file(is_prod=True)
    load_preds(etcbc_fname, outdir)
