"""Load predictions from CSV files."""

import csv
from pathlib import Path
from typing import Literal, TypedDict

from src.classifier.prediction_utils import convert
from src.classifier.result_utils import ProbaPredictions, Verse

class BookVerses(TypedDict):
    """A dict to represent the link between verses and a book."""
    book_name: str
    verses: list[Verse]
    verse_probas: list[list[float]]

def load_preds(
        origin_name: Literal["CAL", "ETCBC"],
        fname: str,
        load_dir: str = "src/classifier/out/",
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
        A list of :class:`src.ui_web.load_predictions.BookVerses` instances
        containing the verses and their prediction results from each book.
    """
    data_origin = origin_name.strip()
    load_dir_p = Path(load_dir)
    parsed_books = []
    books: list[BookVerses] = []

    with (load_dir_p / fname).open(newline="") as csvfile:
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

            if (current_book
                and current_book != row[0]
                and current_book not in parsed_books):
                # when we finish parsing a book
                parsed_books.append(current_book)
                books.append({"book_name": current_book,
                                "verses": verses,
                                "verse_probas": probas
                            })
                # reset the lists
                verses = []
                probas = []

            if (current_book != row[0]):
                # when we discover a new book
                current_book = row[0]


            print(f"(Ref: {row[1]}) OT > {row[2]} NT > {row[3]} [{row[4]}]")
            probas.append([float(row[2]), float(row[3])])

            # initialise the Verse instance based on the data origin
            if data_origin == "CAL":
                verses.append(
                    Verse(
                        row[0],
                        row[1],
                        row[4].split(" "),
                        origin=data_origin
                    )
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
    return books


if __name__ == "__main__":
    outdir = "src/classifier/out/"
    # threshold
    threshold = 0.5

    # CAL
    cal_fname = "PRODUCTION_mnb_cal_prediction_proba_all_both_removed.csv"
    load_preds("CAL", cal_fname, outdir)

    # ETCBC
    etcbc_fname = "PRODUCTION_mnb_etcbc_prediction_proba_all_remove_nonchar.csv"
    load_preds("ETCBC", etcbc_fname, outdir)
