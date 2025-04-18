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

import numpy as np
from numpy.typing import NDArray
from sklearn.naive_bayes import MultinomialNB

from collections.abc import Callable

from src.classifier.dataset_skeleton import DataSplit, LoadedDataset
from src.classifier.eval_utils import (
    eval_and_save,
)
from src.classifier.fname_utils import FnameExtraOpts, SavefileName
from src.classifier.load_cal import load_cal_dataset
from src.classifier.result_utils import Verse
from src.classifier.wrappers import BoWEstimator
from src.shared import label_data


def csvify_cal(
        samples: list[Verse] | NDArray[Verse],
        probas: list[list[float]] | NDArray[np.float64],
        correct_labels: list[int] | NDArray[np.int64] | None = None
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
        csv_data = ("Book,Reference,"
                    + f'"Probability for {label_data.ValToLabel[0]}",'
                    + f'"Probability for {label_data.ValToLabel[1]}",'
                    + "Leammatised Verse,Correct Label\n")

        for i in range(len(probas)):
            csv_data += (
                f"{samples[i].book},{samples[i].reference},{probas[i][0]:.04f},"
                + f"{probas[i][1]:.04f},"
                + f"{' '.join(samples[i].get_translit_words())},"
                + f"{correct_labels[i]}\n"
            )
    else:
        csv_data = ("Book,Reference,"
                    + f'"Probability for {label_data.ValToLabel[0]}",'
                    + f'"Probability for {label_data.ValToLabel[1]}",'
                    + "Leammatised Verse\n")

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
        matches = re.search(r"p\d\d|c", vrs.words[i].annots)
        if matches is not None:
            vrs_lemmata.append(vrs.words[i].translit)
            vrs_annots.append(vrs.words[i].annots)
    if len(vrs_lemmata) != 0:
        return Verse(vrs.book,
                    vrs.reference,
                    vrs_lemmata,
                    words_annotations=vrs_annots,
                    origin="CAL"
                    )
    # else
    return None


def remove_proper_nouns(vrs: Verse) -> Verse | None:
    """Remove proper nouns from the verse.

    Returns:
        Similar to  list of tuples, each containing verse reference,
        lemmata from the verse, annotations for each lemma,
        but without proper nouns and their annotations.

    .. seealso:
        :func:`src.classifier.aa_cal_consistent.remove_proclitic_ubs`
            Removes underscores after proclitics.
    """
    vrs_lemmata = []
    vrs_annots = []
    # for each word in the verse
    for i in range(len(vrs)):
        # look for PN or GN in annots
        match = re.search(r"PN|GN", vrs.words[i].annots)
        if match is None:
            # if a word is not annotated as PN or GN,
            # include the lemma in the training set
            vrs_lemmata.append(vrs.words[i].translit)
            vrs_annots.append(vrs.words[i].annots)
    if len(vrs_lemmata) > 0:
        return Verse(vrs.book, vrs.reference,
                                    vrs_lemmata,
                                    words_annotations=vrs_annots,
                                    origin="CAL"
                                    )
    return None

def cal_eval_classifier(
            clf: BoWEstimator,
            cal_load: LoadedDataset,
            save_f: SavefileName,
            func_to_map: Callable[[Verse], Verse | None] | None = None,
            *,
            map_to_both: bool = False
        ) -> None:
    """Evaluate a classifier with CAL data."""
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


if __name__ == "__main__":
    # load cal data from src/scraper/cal_results
    cal_ds = load_cal_dataset("./src/")
    n_window = 3

    print("\nPlain Classifier")
    print("MultinomialNB")
    c_mnb = BoWEstimator(MultinomialNB(), " ".join, n=n_window)
    save_fname = SavefileName("CAL", "mnb")
    save_fname.set_ngram_opts(n=n_window)

    # train and evaluate
    cal_eval_classifier(c_mnb, cal_ds, save_fname)

    print("\nRemove PN & GN")
    print("> Remove PN & GN from the training set")
    print("MultinomialNB")
    # Remove personal names and place names from the training data
    # and train new classifiers
    c_mnb_r = BoWEstimator(MultinomialNB(), " ".join, n=n_window)
    save_fname_npn = save_fname.copy()
    save_fname_npn.add_extra_opts([FnameExtraOpts.REMOVE_PROPN])
    cal_eval_classifier(c_mnb_r, cal_ds, save_fname_npn, remove_proper_nouns)

    print("\n!!!!!!!!!!!!!!!BELOW REQUIRES DS-wide processing!!!!!!!!!!!!!!!")

    print("\nRemove underscores marking proclitics, "
            + "from both training & test sets")
    print("MultinomialNB")
    c_mnb_nus = BoWEstimator(MultinomialNB(), " ".join, n=n_window)
    c_mnb_nus.set_preprocessor(remove_underscores)

    save_fname_nus = save_fname.copy()
    save_fname_nus.add_extra_opts([FnameExtraOpts.REMOVE_UNDERSCORES,
                                    FnameExtraOpts.REMOVE_FROM_BOTH])

    cal_eval_classifier(c_mnb_nus, cal_ds, save_fname_nus)

    print("\nRemove PN & GN")
    print("> Remove PN & GN from both training & test sets")
    print("MultinomialNB")
    # Remove personal names and place names from
    # both the training and test datasets
    # and train new classifiers
    c_mnb_npn_both = BoWEstimator(MultinomialNB(), " ".join, n=n_window)
    save_fname_npn_both = save_fname_npn.copy()
    save_fname_npn_both.add_extra_opts([FnameExtraOpts.REMOVE_FROM_BOTH])
    cal_eval_classifier(c_mnb_npn_both, cal_ds, save_fname_npn_both,
                        remove_proper_nouns, map_to_both=True)

    print("> Remove PN, GN, underscores from both training & test sets")
    print("MultinomialNB")
    # Remove personal names and place names from
    # both the training and test datasets
    # and train new classifiers
    c_mnb_rnus = BoWEstimator(MultinomialNB(), " ".join, n=n_window)
    c_mnb_rnus.set_preprocessor(remove_underscores)

    save_fname_rnus = save_fname_npn.copy()
    save_fname_rnus.add_extra_opts([FnameExtraOpts.REMOVE_PROPN,
                                FnameExtraOpts.REMOVE_FROM_BOTH])

    cal_eval_classifier(c_mnb_rnus, cal_ds, save_fname_rnus,
                        remove_proper_nouns, map_to_both=True)
