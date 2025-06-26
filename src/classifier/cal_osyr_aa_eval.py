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

"""Authorship attribution using CAL data."""

import re
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from src.classifier.dataset_skeleton import DataSplit, LoadedDataset
from src.classifier.eval_utils import (
    eval_and_save,
)
from src.classifier.fitting_utils import identity
from src.classifier.fname_utils import FnameExtraOpts, SavefileName
from src.classifier.load_cal import load_cal_dataset
from src.classifier.result_utils import Verse
from src.classifier.wrappers import BoWEstimator
from src.shared import label_data
from src.shared.classification_algos import get_algo_by_name


def csvify_cal(
    samples: list[Verse] | NDArray[Verse],
    probas: list[list[float]] | NDArray[np.float64],
    correct_labels: list[int] | NDArray[np.int64] | None = None,
) -> str:
    """Format classifier prediction results into CSV format.

    Args:
        samples: samples that were labelled.
        probas: two-dimensional list of probabilities of each verse
            belonging to each of the classes.
        correct_labels: gold reference for the samples.

    Returns:
        string containing the prediction results in CSV format.
    """
    if len(samples) != len(probas):
        msg = "The number of samples do not match the number of probabilities!"
        raise ValueError(msg)

    if correct_labels is not None:
        csv_data = (
            "Book,Reference,"
            + f'"Probability for {label_data.ValToLabel[0]}",'
            + f'"Probability for {label_data.ValToLabel[1]}",'
            + "Leammatised Verse,Correct Label\n"
        )

        for i in range(len(probas)):
            csv_data += (
                f"{samples[i].book},{samples[i].reference},{probas[i][0]:.04f},"
                + f"{probas[i][1]:.04f},"
                + f"{' '.join(samples[i].get_translit_words())},"
                + f"{correct_labels[i]}\n"
            )
    else:
        csv_data = (
            "Book,Reference,"
            + f'"Probability for {label_data.ValToLabel[0]}",'
            + f'"Probability for {label_data.ValToLabel[1]}",'
            + "Leammatised Verse\n"
        )

        for i in range(len(probas)):
            csv_data += (
                f"{samples[i].book},{samples[i].reference},{probas[i][0]:.04f},"
                + f"{probas[i][1]:.04f},"
                + f"{' '.join(samples[i].get_translit_words())}\n"
            )

    # print(csv_data.split("\n")[1])
    return csv_data


def remove_underscores(words: list[str]) -> list[str]:
    """A preprocessing function to remove the underscores after proclitics.

    Returns:
        a list of str where underscores in all words are removed.
    """
    return [w.replace("_", "") for w in words]


def remove_enclitics(vrs: Verse) -> Verse | None:
    """Remove enclitic prepositions from the text.

    Returns:
        list of tuples, each containing (
        verse reference, lemmata in the verse, annotations for each lemma
        ) where enclitic prepositions are removed.
    """
    vrs_lemmata = []
    vrs_annots = []

    for i in range(len(vrs.words)):
        matches = re.search(r"^p\d\d$|^c$", vrs.words[i].annots)
        if matches is None:
            vrs_lemmata.append(vrs.words[i].translit)
            vrs_annots.append(vrs.words[i].annots)
    if len(vrs_lemmata) != 0:
        return Verse(
            vrs.book,
            vrs.reference,
            vrs_lemmata,
            words_annotations=vrs_annots,
            origin="CAL",
        )
    # else
    return None


def remove_proper_nouns(vrs: Verse) -> Verse | None:
    """Remove proper nouns from the verse.

    Returns:
        a :class:`src.classifier.load_cal.Verse` object without the proper nouns
        or ``None`` if all words are proper nouns.

    .. seealso:
        :func:`src.classifier.aa_cal_consistent.remove_proclitic_ubs`
            Removes underscores after proclitics.
    """
    vrs_lemmata = []
    vrs_annots = []
    vrs_syriac = []
    # for each word in the verse
    for i in range(len(vrs)):
        # look for PN or GN in annots
        match = re.search(r"PN|GN", vrs.words[i].annots)
        if match is None:
            # if a word is not annotated as PN or GN,
            # include the lemma in the training set
            vrs_lemmata.append(vrs.words[i].translit)
            vrs_annots.append(vrs.words[i].annots)
            vrs_syriac.append(vrs.words[i].syriac)
    if len(vrs_lemmata) > 0:
        return Verse(
            vrs.book,
            vrs.reference,
            vrs_lemmata,
            vrs_syriac,
            words_annotations=vrs_annots,
            origin="CAL",
        )
    return None


def cal_eval_classifier(
    clf: BoWEstimator,
    cal_load: LoadedDataset,
    save_f: SavefileName,
    func_to_map: Callable[[Verse], Verse | None] | None = None,
    *,
    map_to_both: bool = False,
) -> None:
    """Evaluate a classifier with the CAL data."""
    train_x = cal_load.train.get_samples()
    train_y = cal_load.train.get_labels()

    if func_to_map is not None:
        train_x = list(map(func_to_map, train_x))
        if map_to_both:
            test_x_ot = list(map(func_to_map, cal_load.test.get_samples(0)))
            test_x_nt = list(map(func_to_map, cal_load.test.get_samples(1)))
            cal_load.test = DataSplit(test_x_ot, test_x_nt)

    clf.fit(train_x, train_y)

    eval_and_save(
        clf,
        cal_load,
        csvify_cal,
        save_f,
        out_dir="./src/classifier/out/",
    )


def run_cal_clf(
    clf_alias: str,
    n_window: int,
    cal_ds: LoadedDataset,
    base_fname: SavefileName,
    **kwargs,
):
    algo = get_algo_by_name(clf_alias, **kwargs)
    print(f"\n================= char {n_window}-gram ===================")
    print("\n> Plain Classifier")
    c_clf = BoWEstimator(algo, " ".join, n=n_window)
    save_fname = base_fname.copy()
    save_fname.set_ngram_opts(n=n_window, is_char_level=True)

    # train and evaluate
    cal_eval_classifier(c_clf, cal_ds, save_fname)

    print("\n> Remove PN & GN")
    print(">> Remove PN & GN from the training set")
    # Remove personal names and place names from the training data
    # and train new classifiers
    c_clf_r = BoWEstimator(algo, " ".join, n=n_window)
    save_fname_npn = save_fname.copy()
    save_fname_npn.add_extra_opts([FnameExtraOpts.REMOVE_PROPN])
    cal_eval_classifier(c_clf_r, cal_ds, save_fname_npn, remove_proper_nouns)

    print("\n!!!!!!BELOW REQUIRES DS-wide processing!!!!!!")

    print(
        "\n> Remove underscores marking proclitics, "
        + "from both training & test sets"
    )
    c_clf_nus = BoWEstimator(algo, " ".join, n=n_window)
    c_clf_nus.set_preprocessor(remove_underscores)

    save_fname_nus = save_fname.copy()
    save_fname_nus.add_extra_opts(
        [FnameExtraOpts.REMOVE_UNDERSCORES, FnameExtraOpts.REMOVE_FROM_BOTH]
    )

    cal_eval_classifier(c_clf_nus, cal_ds, save_fname_nus)

    print("\n> Remove PN & GN")
    print(">> Remove PN & GN from both training & test sets")
    # Remove personal names and place names from
    # both the training and test datasets
    # and train new classifiers
    c_clf_npn_both = BoWEstimator(algo, " ".join, n=n_window)
    save_fname_npn_both = save_fname_npn.copy()
    save_fname_npn_both.add_extra_opts([FnameExtraOpts.REMOVE_FROM_BOTH])
    cal_eval_classifier(
        c_clf_npn_both,
        cal_ds,
        save_fname_npn_both,
        remove_proper_nouns,
        map_to_both=True,
    )

    print("\n>> Remove PN, GN, underscores from both training & test sets")
    # Remove personal names and place names from
    # both the training and test datasets
    # and train new classifiers
    c_clf_rnus = BoWEstimator(algo, " ".join, n=n_window)
    c_clf_rnus.set_preprocessor(remove_underscores)

    save_fname_rnus = save_fname_npn.copy()
    save_fname_rnus.add_extra_opts(
        [FnameExtraOpts.REMOVE_PROPN, FnameExtraOpts.REMOVE_FROM_BOTH]
    )

    cal_eval_classifier(
        c_clf_rnus,
        cal_ds,
        save_fname_rnus,
        remove_proper_nouns,
        map_to_both=True,
    )

    print("\n=============== Word n-grams =================")
    print("\n> Plain Classifier")
    w_clf = BoWEstimator(algo, identity, n=n_window)
    save_fname_w = base_fname.copy()
    save_fname_w.set_ngram_opts(n=n_window, is_char_level=False)

    # train and evaluate
    cal_eval_classifier(w_clf, cal_ds, save_fname_w)

    print("\n> Remove PN & GN")
    print(">> Remove PN & GN from the training set")
    # Remove personal names and place names from the training data
    # and train new classifiers
    w_clf_r = BoWEstimator(algo, identity, n=n_window)
    w_clf_r.set_preprocessor(remove_underscores)

    save_fname_npn_w = save_fname_w.copy()
    save_fname_npn_w.add_extra_opts([FnameExtraOpts.REMOVE_PROPN])
    cal_eval_classifier(w_clf_r, cal_ds, save_fname_npn, remove_proper_nouns)

    print("\n!!!!!!BELOW REQUIRES DS-wide processing!!!!!!")
    print("\n>> Remove PN, GN from both training & test sets")
    # Remove personal names and place names from
    # both the training and test datasets
    # and train new classifiers
    w_clf_rnus = BoWEstimator(algo, identity, n=n_window)
    w_clf_rnus.set_preprocessor(remove_underscores)

    save_fname_rnus_w = save_fname_npn_w.copy()
    save_fname_rnus_w.add_extra_opts(
        [
            FnameExtraOpts.REMOVE_UNDERSCORES,
            FnameExtraOpts.REMOVE_PROPN,
            FnameExtraOpts.REMOVE_FROM_BOTH,
        ]
    )

    cal_eval_classifier(
        w_clf_rnus,
        cal_ds,
        save_fname_rnus_w,
        remove_proper_nouns,
        map_to_both=True,
    )


if __name__ == "__main__":
    # load cal data from src/scraper/cal_results
    cal_ds = load_cal_dataset("./src/")
    # print("Multinomial NB")
    # base_fname_mnb = SavefileName("CAL", "mnb")
    # for n in range(1, 6):
    #     run_cal_clf("mnb", n, cal_ds, base_fname_mnb)

    # print("\n≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈ K-NEAREST NEIGHBOURS ≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈\n")
    # base_fname_knn = SavefileName("CAL", "knn")

    # for n in range(1, 6):
    #     for k in range(2, 10):
    #         print(f"\n> K = {k}")
    #         base_fname_knn.set_knn_opts(k=k)
    #         run_cal_clf("knn", n, cal_ds, base_fname_knn, n_neighbors=k)

    # print("=================== SVC =====================\n")
    # base_fname_svm = SavefileName("CAL", "svc")
    # for n in range(1, 6):
    #     run_cal_clf("svc", n, cal_ds, base_fname_svm, probability=True)

    # print("================ KNN with DISTANCE ==================\n")
    # base_fname_knn = SavefileName("CAL", "knn")

    # for n in range(1, 6):
    #     for k in range(2, 10):
    #         print(f"\n> K = {k}")
    #         base_fname_knn.set_knn_opts(k=k, weights="distance")
    #         run_cal_clf("knn", n, cal_ds, base_fname_knn, n_neighbors=k, weights="distance")

    # print("================== RANDOM FOREST =====================\n")
    # base_fname_rf = SavefileName("CAL", "rf")
    # for n in range(3, 6):
    #     for ne in [100, 200, 300, 400, 500]:
    #         base_fname_rf.set_rf_opts(n_estimators=ne)
    #         run_cal_clf("rf", n, cal_ds, base_fname_rf, n_estimators=ne)

    # print("================== MULTI-LAYER PERCEPTRON =====================\n")
    # base_fname_mlp = SavefileName("CAL", "mlp")
    # for n in range(1, 6):
    #     for n_neurons in [100, 200, 300, 400, 500]:
    #         for i in range(1,3):
    #             hl = [n_neurons] * i
    #             base_fname_mlp.set_mlp_opts(hidden_layer_sizes=hl)
    #             run_cal_clf("mlp", n, cal_ds, base_fname_mlp, hidden_layer_sizes=hl, solver="lbfgs")
    #     run_cal_clf("mlp", n, cal_ds, base_fname_mlp)
