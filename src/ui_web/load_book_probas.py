import csv
from pathlib import Path

outdir = Path("src/classifier/out/")

# CAL
cal_fname = "PRODUCTION_cal_mnb_char_3gram_bow_no_uscore_no_propn_both_removed_total_proba.csv"

cal_book_proba_dict = {}
cal_book_proba_labels = []

with (outdir / cal_fname).open(newline="") as csvfile:
    read_data = csv.reader(csvfile)
    first_row = True
    for row in read_data:
        if first_row:
            first_row = False
            cal_book_proba_labels = row[1:]
            continue
        print(
            f"{row[0]}: (OT) {float(row[1]):.08f} vs. "
            + f"(NT) {float(row[2]):.08f}"
        )
        cal_book_proba_dict.update({row[0]: row[1:]})

# ETCBC
etcbc_fname = "PRODUCTION_etcbc_mnb_char_3gram_bow_total_proba.csv"
etcbc_book_proba_dict = {}
etcbc_book_proba_labels = []

with (outdir / etcbc_fname).open(newline="") as csvfile:
    read_data = csv.reader(csvfile)
    first_row = True
    for row in read_data:
        if first_row:
            first_row = False
            etcbc_book_proba_labels = row[1:]
            continue
        print(
            f"{row[0]}: (OT) {float(row[1]):.08f} vs. "
            + f"(NT) {float(row[2]):.08f}"
        )
        etcbc_book_proba_dict.update({row[0]: row[1:]})
