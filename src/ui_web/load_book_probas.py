import csv
from pathlib import Path
from typing import Literal


class BookProbas:
    """Class to load book probabilities from CSV files.

    Attributes:
        origin: origin of the data (ETCBC or CAL).
        book_proba_dict: dictionary of per-book probabilities.
    """

    def __init__(
            self,
            origin: str
        ) -> None:
        """Load the book probabilities from a CSV file.

        Args:
            origin: origin of the data (ETCBC or CAL).
        """
        self.origin = origin
        # dict of {book_name: [proba for OT, proba for NT]}
        self.book_proba_dict: dict[str, list[float]] = {}

    def add_book_proba(self, book_name: str, proba: list[float]) -> None:
        """Add predicted probabilities of a book to the dictionary.

        Args:
            book_name: name of the book.
            proba: list of probabilities of the book belonging to each class.
        """
        if len(self.book_proba_dict) == 0:
            self.book_proba_dict = {book_name: proba}
        else:
            self.book_proba_dict.update({book_name: proba})


def load_book_probas(
        data_origin: Literal["ETCBC", "CAL"],
        fname: str,
        load_dir: str | Path = "src/classifier/out/",
    ) -> BookProbas:
    """Load total probabilities for books from a CSV file.

    Args:
        data_origin: origin of the data.
        fname: name of the target CSV file.
        load_dir: path to the directory where the CSV file is located.

    Returns:
        A :class:`BookProbas` instance containing the loaded book probabilities.
    """
    book_probas = BookProbas(data_origin)
    load_dir_p = Path(load_dir)

    with (load_dir_p / fname).open(newline="") as csvfile:
        read_data = csv.reader(csvfile)
        first_row = True
        for row in read_data:
            if first_row:
                first_row = False
                continue
            print(f"{row[0]}: (OT) {float(row[1]):.08f} vs."
                    + f" (NT) {float(row[2]):.08f}")
            book_probas.add_book_proba(row[0], [float(row[1]), float(row[2])])
    return book_probas

outdir = Path("src/classifier/out/")

# CAL
print("\nCAL")
cal_fname = "PRODUCTION_cal_mnb_char_3gram_bow_no_uscore_no_propn_both_removed_total_proba.csv"
load_book_probas("CAL", cal_fname, outdir)

# ETCBC
print("\nETCBC")
etcbc_fname = "PRODUCTION_etcbc_mnb_char_3gram_bow_total_proba.csv"
load_book_probas("ETCBC", etcbc_fname, outdir)
