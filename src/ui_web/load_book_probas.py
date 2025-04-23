import csv
from pathlib import Path

from src.classifier.fname_utils import FnameExtraOpts, SavefileName


class BookProbas:
    """Class to load book probabilities from CSV files.

    Attributes:
        origin: origin of the data (ETCBC or CAL).
        book_proba_dict: dictionary of per-book probabilities.
    """

    def __init__(self, origin: str) -> None:
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
    fname: SavefileName,
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
    book_probas = BookProbas(fname.origin)
    fname_p = Path(load_dir) / fname.get_fname()

    with fname_p.open(newline="") as csvfile:
        read_data = csv.reader(csvfile)
        first_row = True
        for row in read_data:
            if first_row:
                first_row = False
            else:
                print(
                    f"{row[0]}: (OT) {float(row[1]):.08f} vs."
                    + f" (NT) {float(row[2]):.08f}"
                )
                book_probas.add_book_proba(
                    row[0], [float(row[1]), float(row[2])]
                )
    return book_probas


if __name__ == "__main__":
    outdir = Path("src/classifier/out/")

    # CAL
    print("\nCAL")
    cal_fname = SavefileName(
        origin="CAL",
        classifier_alias="mnb",
    )
    cal_fname.set_ngram_opts(
        n=3,
        is_n_gram=True,
        is_bow=True,
        is_char_level=True,
    )
    cal_fname.add_extra_opts(
        [
            FnameExtraOpts.REMOVE_UNDERSCORES,
            FnameExtraOpts.REMOVE_PROPN,
            FnameExtraOpts.REMOVE_FROM_BOTH,
        ]
    )
    cal_fname.mark_special_file(is_prod=True, is_total_proba=True)
    load_book_probas(cal_fname, outdir)

    # ETCBC
    print("\nETCBC")
    etcbc_fname = SavefileName(
        origin="ETCBC",
        classifier_alias="mnb",
    )
    etcbc_fname.set_ngram_opts(
        n=3,
        is_n_gram=True,
        is_bow=True,
        is_char_level=True,
    )
    etcbc_fname.add_extra_opts([FnameExtraOpts.REMOVE_DIACRITICS])
    etcbc_fname.mark_special_file(is_prod=True, is_total_proba=True)
    load_book_probas(etcbc_fname, outdir)
