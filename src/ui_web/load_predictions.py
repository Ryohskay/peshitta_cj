"""Load predictions from CSV files."""

import csv

from pathlib import Path
from typing import Literal

from src.classifier.result_utils import ProbaPredictions, Verse
from src.classifier.prediction_utils import convert

def load_preds(
            fname: str,
            load_dir: str = "src/classifier/out/",
            origin_name: Literal["CAL", "ETCBC"] = "CAL",
            threshold: float = 0.5
        ) -> ProbaPredictions:
    """Load prediction results from a CSV file.

    Args:
        fname: name of the target CSV file. You should use the one with "all"
            in the name since such a file contains all predictions on a dataset.
        load_dir: str indicating path to the directory where the CSV file is
            located.
        origin_name: name of the database or dataset(s) the data was originally
            extracted from.
        threshold: threshold of predicted probability to classify a sample to be
            belonging to a particular class.

    Returns:
        A :class:`src.classifier.result_utils.ProbaPredictions` instance
        containing the loaded prediction results.
    """
    load_dir_p = Path(load_dir)
    verses = []
    probas = []
    y = []
    data_origin = origin_name.strip()

    with (load_dir_p / fname).open(newline="") as csvfile:
        read_data = csv.reader(csvfile)
        first_row = True
        for row in read_data:
            if first_row:
                first_row = False
                continue

            print(f"(Ref: {row[1]}) OT > {row[2]} NT > {row[3]} [{row[4]}]")
            probas.append([float(row[2]), float(row[3])])

            if data_origin == "CAL":
                verses.append(Verse(row[0], row[1], row[4].split(" "),
                                    origin=data_origin
                                )
                            )
                if (len(row) > 5):
                    y.append(row[5])
            elif data_origin == "ETCBC":
                print(f"(Ref: {row[1]}) OT > {row[2]} NT > {row[3]} [{row[5]}]")
                verses.append(Verse(row[0], row[1], row[4].split(" "),
                                        syriac_words=row[5].split(" "),
                                        origin=data_origin))
                if (len(row) > 6):
                    y.append(row[6])
    if len(y) == 0:
        y = None

    return ProbaPredictions(verses, convert(probas, threshold), probas, y)

if __name__ == "__main__":
    outdir = "src/classifier/out/"
    # threshold
    threshold = 0.5

    # CAL
    cal_fname = "PRODUCTION_mnb_cal_prediction_proba_all_both_removed.csv"
    load_preds(cal_fname, outdir, "CAL", threshold)


    # ETCBC
    etcbc_fname = "PRODUCTION_mnb_etcbc_prediction_proba_all_remove_nonchar.csv"
    load_preds(etcbc_fname, outdir, "ETCBC", threshold)
