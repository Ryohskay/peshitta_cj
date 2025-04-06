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
from pathlib import Path

from sklearn.naive_bayes import MultinomialNB

from classifier import book_data
from classifier.eval_utils import (
    eval_and_save,
    evaluate_classifier,
)
from classifier.fitting_utils import BoWEstimator
from classifier.load_cal import get_book_verses, load_df_json
from classifier.result_utils import Verse


def csvify_cal(
        samples: list[Verse],
        probabilities: list[list[float]],
        correct_labels: list[int]
    ) -> str:
    """Format classifier prediction results into CSV format.

    Returns:
        string containing the prediction results in CSV format.
    """
    csv_data = ("Reference,Probability for OT,Probability for NT,"
                + "Correct Label,Leammatised Verses")

    for i in range(len(probabilities)):
        csv_data += (
            f"{samples[i].reference},{probabilities[i][0]:.04f},"
            + f"{probabilities[i][1]:.04f},{correct_labels[i]},"
            + f"{samples[i].get_translit_words()}"
        )

    print(csv_data.split("\n")[1])
    return csv_data


def remove_proclitic_ubs(verses: list[Verse]) -> list:
    """Remove the underscores after proclitics.

    Returns:
        list of tuples, each containing:
        (verse reference, lemmata in the verse, annotations for each lemma)
    """
    train_x_no_ub = []
    for vrs in verses:
        vrs_lemmata = vrs.get_translit_words()
        for i in range(len(vrs)):
            if vrs.words[i].annots == "c":
                vrs_lemmata[i] = vrs_lemmata[i].replace("_", "")
        train_x_no_ub.append((vrs.reference, vrs_lemmata, vrs.get_annotations()))
    return train_x_no_ub


def remove_proper_nouns(verses: list[Verse]) -> list:
    """Remove proper nouns from the verse.

    Returns:
        Similar to  list of tuples, each containing:
        (verse reference, lemmata from the verse, annotations for each lemma),
        but without proper nouns and their annotations.

    .. seealso:
        :func:`classifier.aa_cal_consistent.remove_proclitic_ubs`
            Removes underscores after proclitics.
    """
    verses_removed = []
    for vrs in verses:
        # print(vrs)
        vrs_ref = vrs[0]
        vrs_lemmata = []
        vrs_annots = []
        # for each word in the verse
        for i in range(len(vrs[1])):
            # look for PN or GN in annots
            match = re.search(r"PN|GN", vrs[2][i][1])
            if match is None:
                # if a word is not annotated as PN or GN,
                # include the lemma in the training set
                vrs_lemmata.append(vrs[2][i][0])
                vrs_annots.append(vrs[2][i][1])
        if len(vrs_lemmata) > 0:
            verses_removed.append((vrs_ref, vrs_lemmata, vrs_annots))
    return verses_removed


if __name__ == "__main__":
    # load the CSV data files
    # Make sure you don't add slash at the beginning of the file path
    # since it breaks Path concatenation
    PROJ_ROOT = Path("./")
    print(PROJ_ROOT)
    df = load_df_json(PROJ_ROOT / "scraper/cal_results/")

    if df is None:
        msg = f"Failed to fetch data from {PROJ_ROOT / 'scraper/cal_results/'}"
        raise RuntimeError(msg)

    # get the training data
    print("OT_train")
    ot_train_verses = get_book_verses(
        df, book_data.ot_train_books, trim_none=True
    )
    print("NT_train")
    nt_train_verses = get_book_verses(
        df, book_data.nt_train_books, trim_none=True
    )
    train_x = ot_train_verses.copy()
    train_x.extend(nt_train_verses)
    train_y = [0 for v in ot_train_verses]
    train_y.extend([1 for v in nt_train_verses])

    # get the test data
    ot_test_verses = get_book_verses(
        df, book_data.ot_test_books, trim_none=True
    )
    nt_test_verses = get_book_verses(
        df, book_data.nt_test_books, trim_none=True
    )

    # print(train_x[-1])
    # get the production data
    ot_prod = get_book_verses(df, book_data.ot_prod_books, trim_none=True)

    c_mnb = BoWEstimator(MultinomialNB(), " ".join)
    c_mnb.fit(train_x, train_y)

    ot_probas, nt_probas, ot_mislabels, nt_mislabels = evaluate_classifier(
        c_mnb,
        ot_test_verses,
        nt_test_verses,
    )

    ot_mislabels.save_to_file(
        csvify_cal,
        save_file="./out/cal_mnb_prediction_mislabels_ot.csv",
    )
    nt_mislabels.save_to_file(
        formatter=csvify_cal,
        save_file="./out/cal_mnb_prediction_mislabels_nt.csv",
    )

    ot_probas.save_to_file(
        formatter=csvify_cal,
        save_file="./out/cal_mnb_prediction_all_ot.csv",
    )
    nt_probas.save_to_file(
        formatter=csvify_cal,
        save_file="./out/cal_mnb_prediction_all_nt.csv",
    )

    print("Remove underscores marking proclitics, from training set")
    train_x_no_ub = remove_proclitic_ubs(train_x)

    c_mnb_nub = BoWEstimator(MultinomialNB(), " ".join)
    c_mnb_nub.fit(train_x_no_ub, train_y)

    c_mnb_nub, probas_pair_nub = eval_and_save(
                                c_mnb_nub,
                                ot_test_verses,
                                nt_test_verses,
                                csvify_cal,
                                save_file_prefix="cal_",
                                save_file_suffix="_nub"
                            )

    print("Remove underbars marking proclitics, "
            + "from both training & test sets")
    train_x_no_ub = remove_proclitic_ubs(train_x)
    ot_test_verses_no_ub = remove_proclitic_ubs(ot_test_verses)
    nt_test_verses_no_ub = remove_proclitic_ubs(nt_test_verses)

    c_mnb_nub = BoWEstimator(MultinomialNB(), " ".join)
    c_mnb_nub.fit(train_x_no_ub, train_y)

    print("Remove PN & GN")
    print("> Remove PN & GN from the training set")
    # Remove personal names and place names from the training data
    # and train new classifiers
    train_x_removed = remove_proper_nouns(train_x)

    c_mnb_r = BoWEstimator(MultinomialNB(), " ".join)
    c_mnb_r.fit(train_x_removed, train_y)

    c_mnb_r, probas_pair_r = eval_and_save(
                                c_mnb_nub,
                                ot_test_verses,
                                nt_test_verses,
                                csvify_cal,
                                save_file_prefix="cal_",
                                save_file_suffix="_removed"
                            )

    print("Remove PN & GN")
    print("> Remove PN & GN from both training & test sets")
    # Remove personal names and place names from
    # both the training and test datasets
    # and train new classifiers
    ot_test_verses_r = remove_proper_nouns(ot_test_verses)
    nt_test_verses_r = remove_proper_nouns(nt_test_verses)

    c_mnb_rboth = BoWEstimator(MultinomialNB(), " ".join)
    c_mnb_rboth.fit(train_x_removed, train_y)

    c_mnb_r, probas_pair_r = eval_and_save(
                                c_mnb_nub,
                                ot_test_verses,
                                nt_test_verses,
                                csvify_cal,
                                save_file_prefix="cal_",
                                save_file_suffix="_removed_both"
                            )

    print("Remove PN, GN, underbars")
    print("> Remove PN, GN, underbars from both training & test sets")
    # Remove personal names and place names from
    # both the training and test datasets
    # and train new classifiers
    train_x_removed_nub = remove_proper_nouns(train_x_no_ub)
    ot_test_verses_rnub = remove_proper_nouns(ot_test_verses_no_ub)
    nt_test_verses_rnub = remove_proper_nouns(nt_test_verses_no_ub)

    c_mnb_rnub = BoWEstimator(MultinomialNB(), " ".join)
    c_mnb_rnub.fit(train_x_removed_nub, train_y)

    c_mnb_r, probas_pair_r = eval_and_save(
                                c_mnb_nub,
                                ot_test_verses,
                                nt_test_verses,
                                csvify_cal,
                                save_file_prefix="cal_",
                                save_file_suffix="_removed_both_nub"
                            )
