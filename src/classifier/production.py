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

import logging
from pathlib import Path
from pdb import run

from sklearn.naive_bayes import MultinomialNB

from src.classifier.cal_aa_eval import (
    csvify_cal,
    remove_proper_nouns,
    remove_underscores,
)
from src.classifier.dataset_skeleton import LoadedDataset
from src.classifier.etcbc_aa_eval import csvify_etcbc, remove_non_chars
from src.classifier.eval_utils import csvify_total_proba, eval_and_save
from src.classifier.fname_utils import FnameExtraOpts, SavefileName
from src.classifier.load_cal import load_cal_dataset
from src.classifier.prediction_utils import predict_proba
from src.classifier.textfabric_utils import load_etcbc_dataset
from src.classifier.wrappers import BoWEstimator
from src.shared.classification_algos import get_algo_by_name

logger = logging.getLogger(__name__)


def predict_on_prod(
    clf: BoWEstimator,
    loaded_ds: LoadedDataset,
    base_save_fname: SavefileName,
    save_dir: str = "./src/classifier/out/",
    thresh: float = 0.5,
) -> None:
    """Predict on the production data with a given classifier."""
    if base_save_fname.origin == "CAL":
        file_formatter = csvify_cal
    elif base_save_fname.origin == "ETCBC":
        file_formatter = csvify_etcbc
    else:
        msg = f"Unknown dataset origin: {base_save_fname.origin}"
        raise ValueError(msg)

    logger.info(">> Training")
    clf.fit(loaded_ds.train.get_samples(), loaded_ds.train.get_labels())

    logger.info(">> Quick Evaluation")
    save_fname = base_save_fname.copy()

    eval_and_save(
        clf,
        loaded_ds,
        file_formatter,
        save_fname,
        out_dir=save_dir,
        threshold=thresh,
    )

    logger.info(">> Production Data")
    preds = predict_proba(clf, loaded_ds.production, threshold=thresh)

    # save the predictions on the production data
    save_proba_fname = save_fname.copy()
    save_proba_fname.mark_special_file(is_prod=True)
    save_dir_p = Path(save_dir)
    savefile_p = save_dir_p / save_proba_fname.get_fname()
    # write to the save file
    msg = f"Saving predictions to {savefile_p.resolve()}"
    logger.info(msg)
    preds.save_to_file(file_formatter, savefile_p)

    # Save per-book total probas
    save_dir_p = Path(save_dir)
    save_total_proba_fname = save_fname.copy()
    save_total_proba_fname.mark_special_file(is_prod=True, is_total_proba=True)
    total_proba_save_fp = save_dir_p / save_total_proba_fname.get_fname()
    msg = f"Saving total probabilities to {total_proba_save_fp.resolve()}"
    logger.info(msg)
    total_proba_save_fp.write_text(csvify_total_proba(preds.get_total_probas()))

def run_prod_clf(  # noqa: PLR0913
    clf_alias: str,
    n_window: int,
    etcbc_ds: LoadedDataset,
    cal_ds: LoadedDataset,
    base_fname_etc: SavefileName,
    base_fname_cal: SavefileName,
    **kwargs
):
    print("ETCBC --->")
    algo = get_algo_by_name(clf_alias, **kwargs)
    print("> Plain Classifier")
    c_clf = BoWEstimator(algo, " ".join, n=n_window)
    c_clf.set_preprocessor(remove_non_chars)
    c_clf_fname = base_fname_etc.copy()
    print(f"{c_clf_fname.origin}")
    c_clf_fname.set_ngram_opts(n=c_clf.n)
    c_clf_fname.add_extra_opts([FnameExtraOpts.REMOVE_DIACRITICS])
    predict_on_prod(c_clf, etcbc_ds, c_clf_fname, thresh=thresh)

    print("CAL --->")
    print("> Plain Classifier")
    c_clf = BoWEstimator(algo, " ".join, n=n_window)
    c_clf_fname = base_fname_cal.copy()
    c_clf_fname.set_ngram_opts(n=c_clf.n)
    predict_on_prod(c_clf, cal_ds, c_clf_fname, thresh=thresh)

    print("> Remove PN, GN & underscores from the training & test set")
    # Remove personal names and place names from the training data
    # and train new classifiers
    c_clf_r = BoWEstimator(algo, " ".join)
    c_clf_r.set_preprocessor(remove_underscores)
    c_clf_r_fname = c_clf_fname.copy()
    c_clf_r_fname.add_extra_opts(
        [
            FnameExtraOpts.REMOVE_UNDERSCORES,
            FnameExtraOpts.REMOVE_PROPN,
            FnameExtraOpts.REMOVE_FROM_BOTH,
        ]
    )
    cal_ds.test.map_on_samples(remove_proper_nouns)
    predict_on_prod(c_clf_r, cal_ds, c_clf_r_fname, thresh=thresh)

if __name__ == "__main__":
    thresh = 0.9  # probability threshold
    n = 3  # n of n-gram

    etcbc_loaded = load_etcbc_dataset()
    cal_loaded = load_cal_dataset("./src")

    print("MultinomialNB")
    fname_etcbc = SavefileName("ETCBC", "mnb")
    fname_cal = SavefileName("ETCBC", "mnb")
    run_prod_clf("mnb", n, etcbc_loaded, cal_loaded, fname_etcbc, fname_cal)

