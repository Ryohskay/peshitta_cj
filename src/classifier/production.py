# BSD 2-Clause License
#
# Copyright (c) 2025, Ryosuke Nagata
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
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

from pathlib import Path

from sklearn.naive_bayes import MultinomialNB

from src.classifier.cal_aa_eval import (
    csvify_cal,
    remove_proper_nouns,
    remove_underscores,
)
from src.classifier.etcbc_aa_eval import csvify_etcbc, remove_non_chars
from src.classifier.eval_utils import eval_and_save
from src.classifier.load_cal import load_cal_dataset
from src.classifier.prediction_utils import predict_proba
from src.classifier.textfabric_utils import load_etcbc_dataset
from src.classifier.wrappers import BoWEstimator
from src.classifier.fname_utils import SavefileName, FnameExtraOpts
from src.classifier.result_utils import FileFormatterProto
from src.classifier.dataset_skeleton import LoadedDataset

def csvify_total_proba(total_proba_dict: dict) -> str:
    """Format the total proba dict data into CSV.

    Returns:
        the total probability of books as a CSV-formatted str.
    """
    total_proba_csv = "Book,Probability for OT,Probability for NT\n"

    for prod_book in total_proba_dict:
        print(f"{prod_book}: (OT) {total_proba_dict[prod_book][0]}, "
                + f"(NT) {total_proba_dict[prod_book][1]}")
        total_proba_csv += f"{prod_book},{total_proba_dict[prod_book][0]},{total_proba_dict[prod_book][1]}\n"
    return total_proba_csv

def predict_on_prod(
        clf: BoWEstimator,
        loaded_ds: LoadedDataset,
        file_formatter: FileFormatterProto,
        save_dir: str = "./src/classifier/out/",
        thresh: float = 0.5
    ):
    """Predict on the production data with a given classifier."""

    train_x = remove_non_chars(loaded_ds.train.get_samples())
    train_y = loaded_ds.train.get_labels()

    print(">> Quick Evaluation")
    save_f = SavefileName("ETCBC", "mnb")
    save_f.set_ngram_opts(n)
    eval_and_save(clf, loaded_ds, file_formatter,
                save_f,
                out_dir=save_dir,
                save_file_prefix="just_to_check_", threshold=thresh)

    print(">> Production Data")
    ot_prod = remove_non_chars(etcbc_loaded.production)

    preds = predict_proba(c_mnb, etcbc_loaded.production,
                            threshold=thresh)

    save_file = Path("./src/classifier/out/PRODUCTION_mnb_etcbc_"
                    + "prediction_proba_all_remove_nonchar.csv")
    preds.save_to_file(csvify_etcbc, save_file)

    # Save proba_dict
    save_f = Path(save_file.parent / "PRODUCTION_mnb_cal_book_total_proba_remove_nonchar.csv")
    save_f.write_text(csvify_total_proba(preds.get_total_probas()))

if __name__ == "__main__":
    # probability threshold
    thresh = 0.9
    n = 3

    # LOAD ETCBC DATA
    etcbc_loaded = load_etcbc_dataset()
    train_x_etc = remove_non_chars(etcbc_loaded.train.get_samples())
    train_y_etc = etcbc_loaded.train.get_labels()
    print("ETCBC --->")
    print("MultinomialNB")
    print("> Plain Classifier")
    c_mnb = BoWEstimator(MultinomialNB(), " ".join)
    print(">> Training")
    c_mnb.fit(train_x_etc, train_y_etc)


    print("> Character unigram classifier")
    print(">> Training")
    c_mnb = BoWEstimator(MultinomialNB(), " ".join, n=1)
    c_mnb.fit(train_x_etc, train_y_etc)
    print(">> Quick Evaluation")
    eval_and_save(c_mnb, etcbc_loaded, csvify_cal,
                    out_dir="./src/classifier/out/",
                    save_file_prefix="just_to_check_", threshold=thresh)

    print("Production Data")
    ot_prod = remove_non_chars(etcbc_loaded.production)

    preds = predict_proba(c_mnb, etcbc_loaded.production, threshold=thresh)
    total_proba_dict = preds.get_total_probas()
    total_proba_csv = "Book,Probability for OT,Probability for NT\n"

    for prod_book in total_proba_dict:
        print(f"{prod_book}: (OT) {total_proba_dict[prod_book][0]}, "
                + f"(NT) {total_proba_dict[prod_book][1]}")
        total_proba_csv += f"{prod_book},{total_proba_dict[prod_book][0]},{total_proba_dict[prod_book][1]}\n"

    save_file = Path("./src/classifier/out/PRODUCTION_mnb_etcbc_"
                        + "prediction_proba_all_char_unigram.csv")
    preds.save_to_file(csvify_etcbc, save_file)
    # Save proba_dict
    save_f = Path(save_file.parent / "PRODUCTION_mnb_etcbc_book_total_proba_char_unigram.csv")
    save_f.write_text(total_proba_csv)

    print("CAL --->")
    # LOAD CAL DATA
    cal_loaded = load_cal_dataset("./src")

    train_x_cal = cal_loaded.train.get_samples()
    train_y_cal = cal_loaded.train.get_labels()
    print("MultinomialNB")
    print("> Plain Classifier")
    print(">> Training")
    c_mnb = BoWEstimator(MultinomialNB(), " ".join)
    c_mnb.fit(train_x_cal, train_y_cal)
    print(">> Quick Evaluation")
    eval_and_save(c_mnb, cal_loaded, csvify_cal,
                    out_dir="./src/classifier/out/",
                    save_file_prefix="just_to_check_", threshold=thresh)

    print("Production Data")
    ot_prod = cal_loaded.production

    preds = predict_proba(c_mnb, cal_loaded.production, threshold=thresh)
    total_proba_dict = preds.get_total_probas()
    total_proba_csv = "Book,Probability for OT,Probability for NT\n"
    for prod_book in total_proba_dict:
        print(f"{prod_book}: (OT) {total_proba_dict[prod_book][0]}, (NT) {total_proba_dict[prod_book][1]}")
        total_proba_csv += f"{prod_book},{total_proba_dict[prod_book][0]},{total_proba_dict[prod_book][1]}\n"

    save_file = Path("./src/classifier/out/PRODUCTION_mnb_cal_"
                        + "prediction_proba_all.csv")
    preds.save_to_file(csvify_cal, save_file)
    # Save proba_dict
    save_f = Path(save_file.parent / "PRODUCTION_mnb_cal_book_total_proba.csv")
    save_f.write_text(total_proba_csv)

    print("> Remove PN, GN & underscores from the training & test set")
    print(">> Training")
    # Remove personal names and place names from the training data
    # and train new classifiers
    train_x_removed = remove_proper_nouns(remove_underscores(train_x_cal))

    c_mnb_r = BoWEstimator(MultinomialNB(), " ".join)
    c_mnb_r.fit(train_x_removed, train_y_cal)

    print(">> Quick Evaluation")
    eval_and_save(c_mnb_r, cal_loaded, csvify_cal,

                    out_dir="./src/classifier/out/",
                    threshold=thresh)

    print("Production Data")
    ot_prod = remove_proper_nouns(remove_underscores(cal_loaded.production))

    preds = predict_proba(c_mnb, ot_prod, threshold=thresh)
    total_proba_dict = preds.get_total_probas()
    total_proba_csv = "Book,Probability for OT,Probability for NT\n"
    for prod_book in total_proba_dict:
        print(f"{prod_book}: (OT) {total_proba_dict[prod_book][0]}, "
                + f"(NT) {total_proba_dict[prod_book][1]}")
        total_proba_csv += f"{prod_book},{total_proba_dict[prod_book][0]},{total_proba_dict[prod_book][1]}\n"

    save_file = Path("./src/classifier/out/PRODUCTION_mnb_cal_"
                        + "prediction_proba_all_both_removed.csv")
    preds.save_to_file(csvify_cal, save_file)

    # Save proba_dict
    save_f = Path(save_file.parent / "PRODUCTION_mnb_cal_book_total_proba_both_removed.csv")
    save_f.write_text(total_proba_csv)
