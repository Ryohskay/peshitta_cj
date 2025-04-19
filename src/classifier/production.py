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
from typing import Literal

from sklearn.naive_bayes import MultinomialNB

from src.classifier.cal_aa_eval import (
    csvify_cal,
    remove_proper_nouns,
    remove_underscores,
)
from src.classifier.etcbc_aa_eval import csvify_etcbc, remove_non_chars
from src.classifier.eval_utils import eval_and_save, csvify_total_proba
from src.classifier.load_cal import load_cal_dataset
from src.classifier.prediction_utils import predict_proba
from src.classifier.textfabric_utils import load_etcbc_dataset
from src.classifier.wrappers import BoWEstimator, ProbaClassifier
from src.classifier.fname_utils import SavefileName, FnameExtraOpts
from src.classifier.result_utils import FileFormatterProto
from src.classifier.dataset_skeleton import LoadedDataset

def predict_on_prod(  # noqa: PLR0913
        clf: BoWEstimator,
        loaded_ds: LoadedDataset,
        base_save_fname: SavefileName,
        save_dir: str = "./src/classifier/out/",
        thresh: float = 0.5
    ) -> None:
    """Predict on the production data with a given classifier."""
    if base_save_fname.origin == "CAL":
        file_formatter = csvify_cal
    elif base_save_fname.origin == "ETCBC":
        file_formatter = csvify_etcbc

    print(">> Training")
    clf.fit(loaded_ds.train.get_samples(), loaded_ds.train.get_labels())

    print(">> Quick Evaluation")
    save_fname = base_save_fname.copy()

    eval_and_save(clf,
                    loaded_ds,
                    file_formatter,
                    save_fname,
                    out_dir=save_dir,
                    threshold=thresh
                )

    print(">> Production Data")
    preds = predict_proba(c_mnb, loaded_ds.production,
                            threshold=thresh)
    # save the predictions on the production data
    save_proba_fname = save_fname.copy()
    save_proba_fname.mark_special_file(is_prod=True)
    preds.save_to_file(csvify_etcbc, save_proba_fname)

    # Save per-book total probas
    save_dir_p = Path(save_dir)
    save_total_proba_fname = save_fname.copy()
    save_total_proba_fname.mark_special_file(is_prod=True, is_total_proba=True)
    total_proba_save_fp = save_dir_p / save_total_proba_fname.get_fname()
    total_proba_save_fp.write_text(csvify_total_proba(preds.get_total_probas()))


if __name__ == "__main__":
    thresh = 0.9  # probability threshold
    n_window = 3  # n of n-gram

    # LOAD ETCBC DATA
    etcbc_loaded = load_etcbc_dataset()
    print("ETCBC --->")
    print("MultinomialNB")
    print("> Plain Classifier")
    c_mnb = BoWEstimator(MultinomialNB(), " ".join, n=n_window)
    c_mnb.set_preprocessor(remove_non_chars)
    c_mnb_fname = SavefileName("ETCBC", "mnb")
    c_mnb_fname.set_ngram_opts(n=c_mnb.n)
    predict_on_prod(c_mnb, etcbc_loaded, c_mnb_fname, thresh=thresh)

    print("> Character unigram classifier")
    cu_mnb = BoWEstimator(MultinomialNB(), " ".join, n=1)
    cu_mnb.set_preprocessor(remove_non_chars)
    cu_mnb_fname = SavefileName("ETCBC", "mnb")
    cu_mnb_fname.set_ngram_opts(n=cu_mnb.n)
    predict_on_prod(c_mnb, etcbc_loaded, cu_mnb_fname, thresh=thresh)

    print("CAL --->")
    # LOAD CAL DATA
    cal_loaded = load_cal_dataset("./src")

    train_x_cal = cal_loaded.train.get_samples()
    train_y_cal = cal_loaded.train.get_labels()
    print("MultinomialNB")
    print("> Plain Classifier")
    c_mnb = BoWEstimator(MultinomialNB(), " ".join, n=n_window)
    c_mnb_fname = SavefileName("CAL", "mnb")
    c_mnb_fname.set_ngram_opts(n=c_mnb.n)
    predict_on_prod(c_mnb, cal_loaded, c_mnb_fname, thresh=thresh)

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
