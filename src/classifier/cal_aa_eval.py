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

from classifier.dataset_skeleton import DataSplit
from classifier.eval_utils import (
    eval_and_save,
)
from classifier.load_cal import load_cal_dataset
from classifier.result_utils import Verse
from classifier.wrappers import BoWEstimator


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
    if correct_labels is not None:
        csv_data = ("Reference,Probability for OT,Probability for NT,"
                    + "Correct Label,Leammatised Verse\n")

        for i in range(len(probas)):
            csv_data += (
                f"{samples[i].reference},{probas[i][0]:.04f},"
                + f"{probas[i][1]:.04f},{correct_labels[i]},"
                + f"{' '.join(samples[i].get_translit_words())}\n"
            )
    else:
        csv_data = ("Reference,Probability for OT,Probability for NT,"
                    + "Leammatised Verse\n")

        for i in range(len(probas)):
            csv_data += (
                f"{samples[i].reference},{probas[i][0]:.04f},"
                + f"{probas[i][1]:.04f},"
                + f"{' '.join(samples[i].get_translit_words())}\n"
            )

    # print(csv_data.split("\n")[1])
    return csv_data


def remove_underscores(verses: list[Verse]) -> list[Verse]:
    """Remove the underscores after proclitics.

    Returns:
        list of tuples, each containing (
        verse reference, lemmata in the verse, annotations for each lemma
        ) where underscores are removed.
    """
    train_x_no_ub = []
    for vrs in verses:
        vrs_lemmata = []

        for i in range(len(vrs.words)):
            w = vrs.words[i].translit
            vrs_lemmata.append(w.replace("_", ""))

        updated_verse = Verse(vrs.book,
                                   vrs.reference,
                                   vrs_lemmata,
                                   words_annotations=vrs.get_annotations(),
                                   origin="CAL"
                                   )
        train_x_no_ub.append(updated_verse)
    return train_x_no_ub


def remove_enclitics(verses: list[Verse]) -> list[Verse]:
    """Remove enclitic prepositions from the text.

    Returns:
        list of tuples, each containing (
        verse reference, lemmata in the verse, annotations for each lemma
        ) where enclitic prepositions are removed.
    """
    train_x_no_ub = []
    for vrs in verses:
        vrs_lemmata = []
        vrs_annots = []

        for i in range(len(vrs.words)):
            matches = re.search(r"p\d\d", vrs.words[i].annots)
            if matches is not None:
                vrs_lemmata.append(vrs.words[i].translit)
                vrs_annots.append(vrs.words[i].annots)
        updated_verse = Verse(vrs.book,
                                   vrs.reference,
                                   vrs_lemmata,
                                   words_annotations=vrs_annots,
                                   origin="CAL"
                                   )
        if updated_verse != vrs:
            print(f"vrs {vrs}")
        train_x_no_ub.append(updated_verse)
    return train_x_no_ub


def remove_proper_nouns(verses: list[Verse]) -> list[Verse]:
    """Remove proper nouns from the verse.

    Returns:
        Similar to  list of tuples, each containing verse reference,
        lemmata from the verse, annotations for each lemma,
        but without proper nouns and their annotations.

    .. seealso:
        :func:`classifier.aa_cal_consistent.remove_proclitic_ubs`
            Removes underscores after proclitics.
    """
    # print(f"Target: {verses[0]}")
    verses_trimmed = []
    for vrs in verses:
        # print(vrs)
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
            verses_trimmed.append(Verse(vrs.book, vrs.reference,
                                        vrs_lemmata,
                                        words_annotations=vrs_annots,
                                        origin="CAL"
                                        )
                                  )
    return verses_trimmed


if __name__ == "__main__":
    cal_ds = load_cal_dataset("./")

    train_x = cal_ds.train.get_samples()
    train_y = cal_ds.train.get_labels()

    print("\nPlain Classifier")
    print("MultinomialNB")
    c_mnb = BoWEstimator(MultinomialNB(), "".join)
    c_mnb.fit(train_x, train_y)

    # train and evaluate
    eval_and_save(
                c_mnb,
                cal_ds,
                csvify_cal,
                out_dir="./classifier/out/",
                save_file_prefix="cal_mnb_char_",
            )

    print("\nRemove underscores marking proclitics, from training set")
    print("MultinomialNB")
    train_x_no_ub = remove_underscores(train_x)

    c_mnb_nub = BoWEstimator(MultinomialNB(), "".join)
    c_mnb_nub.fit(train_x_no_ub, train_y)

    c_mnb_nub, probas_pair_nub = eval_and_save(
                                c_mnb_nub,
                                cal_ds,
                                csvify_cal,
                                out_dir="./classifier/out/",
                                save_file_prefix="cal_mnb_char_",
                                save_file_suffix="_nub"
                            )

    print("\nRemove PN & GN")
    print("> Remove PN & GN from the training set")
    print("MultinomialNB")
    # Remove personal names and place names from the training data
    # and train new classifiers
    train_x_removed = remove_proper_nouns(train_x)

    c_mnb_r = BoWEstimator(MultinomialNB(), "".join)
    c_mnb_r.fit(train_x_removed, train_y)

    c_mnb_r, probas_pair_r = eval_and_save(
                                c_mnb_r,
                                cal_ds,
                                csvify_cal,
                                out_dir="./classifier/out/",
                                save_file_prefix="cal_mnb_char_",
                                save_file_suffix="_removed"
                            )

    print("\nRemove PN, GN, underscores")
    print("> Remove PN, GN, underscores from training set")
    print("MultinomialNB")
    # Remove personal names and place names from
    # both the training and test datasets
    # and train new classifiers
    train_x_removed_nub = remove_proper_nouns(train_x_no_ub)

    # for j in range(10):
    #     print(train_x_removed_nub[j])
    c_mnb_rnub = BoWEstimator(MultinomialNB(), "".join)
    c_mnb_rnub.fit(train_x_removed_nub, train_y)

    c_mnb_rnub, probas_pair_rnub = eval_and_save(
                                c_mnb_rnub,
                                cal_ds,
                                csvify_cal,
                                out_dir="./classifier/out/",
                                save_file_prefix="cal_mnb_char_",
                                save_file_suffix="_removed_nub"
                            )

    print("\n!!!!!!!!!!!!!!!BELOW REQUIRES DS-wide processing!!!!!!!!!!!!!!!")
    ot_test_verses = cal_ds.test.get_samples(0)
    nt_test_verses = cal_ds.test.get_samples(1)
    print("\nRemove underscores marking proclitics, "
            + "from both training & test sets")
    print("MultinomialNB")
    ot_test_verses_no_ub = remove_underscores(ot_test_verses)
    nt_test_verses_no_ub = remove_underscores(nt_test_verses)

    cal_ds.test = DataSplit(ot_test_verses_no_ub, nt_test_verses_no_ub)

    c_mnb_nub_both = BoWEstimator(MultinomialNB(), "".join)
    c_mnb_nub_both.fit(train_x_no_ub, train_y)

    c_mnb_nub_both, probas_pair_nub_both = eval_and_save(
                                c_mnb_nub_both,
                                cal_ds,
                                csvify_cal,
                                out_dir="./classifier/out/",
                                save_file_prefix="cal_mnb_char_",
                                save_file_suffix="_nub"
                            )

    print("\nRemove PN & GN")
    print("> Remove PN & GN from both training & test sets")
    print("MultinomialNB")
    # Remove personal names and place names from
    # both the training and test datasets
    # and train new classifiers
    ot_test_verses_r = remove_proper_nouns(ot_test_verses)
    nt_test_verses_r = remove_proper_nouns(nt_test_verses)

    cal_ds.test = DataSplit(ot_test_verses_r, nt_test_verses_r)

    c_mnb_rboth = BoWEstimator(MultinomialNB(), "".join)
    c_mnb_rboth.fit(train_x_removed, train_y)

    c_mnb_rboth, probas_pair_rboth = eval_and_save(
                                c_mnb_rboth,
                                cal_ds,
                                csvify_cal,
                                out_dir="./classifier/out/",
                                save_file_prefix="cal_mnb_char_",
                                save_file_suffix="_removed_both"
                            )

    print("\nRemove PN, GN, underscores")
    print("> Remove PN, GN, underscores from both training & test sets")
    print("MultinomialNB")
    # Remove personal names and place names from
    # both the training and test datasets
    # and train new classifiers
    train_x_removed_both_nub = remove_proper_nouns(train_x_no_ub)
    ot_test_verses_rboth_nub = remove_proper_nouns(ot_test_verses_no_ub)
    nt_test_verses_rboth_nub = remove_proper_nouns(nt_test_verses_no_ub)

    cal_ds.test = DataSplit(ot_test_verses_rboth_nub, nt_test_verses_rboth_nub)
    # print(list(map(str, cal_ds.test.get_samples()[:10])))

    c_mnb_rboth_nub = BoWEstimator(MultinomialNB(), "".join)
    c_mnb_rboth_nub.fit(train_x_removed_both_nub, train_y)

    c_mnb_rboth_nub, probas_pair_rboth_nub = eval_and_save(
                                c_mnb_rboth_nub,
                                cal_ds,
                                csvify_cal,
                                out_dir="./classifier/out/",
                                save_file_prefix="cal_mnb_char_",
                                save_file_suffix="_removed_both_nub"
                            )
