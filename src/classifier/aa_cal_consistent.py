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

import re

from sklearn.naive_bayes import MultinomialNB

from classifier import book_data
from classifier.eval_utils import (
    evaluate_classifier,
    save_all_preds,
    save_mislabels,
)
from classifier.fitting_utils import BoW_Estimator
from classifier.load_cal import get_book_verses, load_df_json


def csvify_cal(
    verses_x: list[str],
    probas: list[list[float]],
    y_correct: list[int],
) -> str:
    """Format classifier prediction results into CSV format."""
    csv_data = ("Reference,Probability for OT,Probability for NT,"
                + "Correct Label,Leammatised Verses")

    if len(verses_x) != len(probas) or len(y_correct) != len(probas):
        raise ValueError

    for i in range(len(probas)):
        csv_data += (
            f"{verses_x[i][0]},{probas[i][0]:.04f},"
            + f"{probas[i][1]:.04f},{y_correct[i]},"
            + f"{' '.join(verses_x[i][1])}"
        )

    # print(csv_data.split('')[1])
    return csv_data


def remove_proclitic_ubs(verses: list) -> list:
    train_x_no_ub = []
    for vrs in verses:
        vrs_lemmata = []
        for i in range(len(vrs[1])):
            if vrs[2][i][1] == "c":
                vrs_lemmata.append(vrs[1][i][0].replace("_", ""))
            else:
                vrs_lemmata.append(vrs[1][i][0])
        train_x_no_ub.append((vrs[0], vrs_lemmata, vrs[2]))
    return train_x_no_ub


def remove_proper_nouns(verses: list) -> list:
    verses_removed = []
    for vrs in verses:
        # print(vrs)
        vrs_ref = vrs[0]
        vrs_lemmata = []
        vrs_annots = []
        # for each word in the verse
        for i in range(len(vrs[1])):
            # look for PN or GN in annots
            match = re.search("PN|GN", vrs[2][i][1])
            if match is None:
                # if a word is not annotated as PN or GN,
                # include the lemma in the training set
                vrs_lemmata.append(vrs[2][i][0])
                vrs_annots.append(vrs[2][i][1])
        if len(vrs_lemmata) > 0:
            verses_removed.append((vrs_ref, vrs_lemmata, vrs_annots))
    return verses_removed


if __name__ == "__main__":
    df = load_df_json(PROJ_ROOT / "/scraper/cal_results/")

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
    train_y = [0 for i in range(len(ot_train_verses))]
    train_y.extend([1 for i in range(len(nt_train_verses))])

    ot_test_verses = get_book_verses(
        df, book_data.ot_test_books, trim_none=True
    )
    nt_test_verses = get_book_verses(
        df, book_data.nt_test_books, trim_none=True
    )
    ot_test_labels = [0 for i in range(len(ot_test_verses))]
    nt_test_labels = [1 for i in range(len(nt_test_verses))]

    # print(train_x[-1])

    ot_prod = get_book_verses(df, book_data.ot_prod_books, trim_none=True)

    c_mnb = BoW_Estimator(MultinomialNB(), " ".join)
    c_mnb.fit(train_x, train_y)

    ot_probas, nt_probas, ot_mislabels, nt_mislabels = evaluate_classifier(
        c_mnb,
        ot_test_verses,
        ot_test_labels,
        nt_test_verses,
        nt_test_labels,
    )

    save_mislabels(
        ot_mislabels,
        nt_mislabels,
        formatter=csvify_cal,
        ot_save_file="./out/cal_mnb_prediction_mislabels_ot.csv",
        nt_save_file="./out/cal_mnb_prediction_mislabels_nt.csv",
    )
    save_all_preds(
        ot_test_verses,
        ot_probas,
        ot_test_labels,
        nt_test_verses,
        nt_probas,
        nt_test_labels,
        formatter=csvify_cal,
        ot_save_file="./out/cal_mnb_prediction_all_ot.csv",
        nt_save_file="./out/cal_mnb_prediction_all_nt.csv",
    )

    print("Remove underscores marking proclitics, from training set")
    train_x_no_ub = remove_proclitic_ubs(train_x)

    c_mnb_nub = BoW_Estimator(MultinomialNB(), " ".join)
    c_mnb_nub.fit(train_x_no_ub, train_y)

    (ot_probas_nub, nt_probas_nub, ot_mislabels_nub, nt_mislabels_nub) = (
        evaluate_classifier(
            c_mnb_nub,
            ot_test_verses,
            ot_test_labels,
            nt_test_verses,
            nt_test_labels,
        )
    )

    save_mislabels(
        ot_mislabels_nub,
        nt_mislabels_nub,
        formatter=csvify_cal,
        ot_save_file="./out/cal_mnb_prediction_mislabels_ot_nub.csv",
        nt_save_file="./out/cal_mnb_prediction_mislabels_nt_nub.csv",
    )

    save_all_preds(
        ot_test_verses,
        ot_probas_nub,
        ot_test_labels,
        nt_test_verses,
        nt_probas_nub,
        nt_test_labels,
        formatter=csvify_cal,
        ot_save_file="./out/cal_mnb_prediction_all_ot_nub.csv",
        nt_save_file="./out/cal_mnb_prediction_all_nt_nub.csv",
    )

    print("Remove underbars marking proclitics, "
            + "from both training & test sets")
    train_x_no_ub = remove_proclitic_ubs(train_x)
    ot_test_verses_no_ub = remove_proclitic_ubs(ot_test_verses)
    nt_test_verses_no_ub = remove_proclitic_ubs(nt_test_verses)

    c_mnb_nub = BoW_Estimator(MultinomialNB(), " ".join)
    c_mnb_nub.fit(train_x_no_ub, train_y)

    print("Remove PN & GN")
    print("> Remove PN & GN from the training set")
    # Remove personal names and place names from the training data
    # and train new classifiers
    train_x_removed = remove_proper_nouns(train_x)

    c_mnb_r = BoW_Estimator(MultinomialNB(), " ".join)
    c_mnb_r.fit(train_x_removed, train_y)

    (ot_probas_r, nt_probas_r, ot_mislabels_r, nt_mislabels_r) = (
        evaluate_classifier(
            c_mnb_r,
            ot_test_verses,
            ot_test_labels,
            nt_test_verses,
            nt_test_labels,
        )
    )

    save_mislabels(
        ot_mislabels_r,
        nt_mislabels_r,
        formatter=csvify_cal,
        ot_save_file="./out/cal_mnb_prediction_mislabels_ot_removed.csv",
        nt_save_file="./out/cal_mnb_prediction_mislabels_nt_removed.csv",
    )

    save_all_preds(
        ot_test_verses,
        ot_probas_r,
        ot_test_labels,
        nt_test_verses,
        nt_probas_r,
        nt_test_labels,
        formatter=csvify_cal,
        ot_save_file="./out/cal_mnb_prediction_all_ot_removed.csv",
        nt_save_file="./out/cal_mnb_prediction_all_nt_removed.csv",
    )

    print("Remove PN & GN")
    print("> Remove PN & GN from both training & test sets")
    # Remove personal names and place names from
    # both the training and test datasets
    # and train new classifiers
    ot_test_verses_r = remove_proper_nouns(ot_test_verses)
    nt_test_verses_r = remove_proper_nouns(nt_test_verses)

    c_mnb_rboth = BoW_Estimator(MultinomialNB(), " ".join)
    c_mnb_rboth.fit(train_x_removed, train_y)

    (
        ot_probas_rboth,
        nt_probas_rboth,
        ot_mislabels_rboth,
        nt_mislabels_rboth,
    ) = evaluate_classifier(
        c_mnb_rboth,
        ot_test_verses_r,
        ot_test_labels,
        nt_test_verses_r,
        nt_test_labels,
    )

    save_mislabels(
        ot_mislabels_rboth,
        nt_mislabels_rboth,
        formatter=csvify_cal,
        ot_save_file="./out/cal_mnb_prediction_mislabels_ot_removed_both.csv",
        nt_save_file="./out/cal_mnb_prediction_mislabels_nt_removed_both.csv",
    )

    save_all_preds(
        ot_test_verses_r,
        ot_probas_rboth,
        ot_test_labels,
        nt_test_verses_r,
        nt_probas_rboth,
        nt_test_labels,
        formatter=csvify_cal,
        ot_save_file="./out/cal_mnb_prediction_all_ot_removed_both.csv",
        nt_save_file="./out/cal_mnb_prediction_all_nt_removed_both.csv",
    )

    print("Remove PN, GN, underbars")
    print("> Remove PN, GN, underbars from both training & test sets")
    # Remove personal names and place names from
    # both the training and test datasets
    # and train new classifiers
    train_x_removed_nub = remove_proper_nouns(train_x_no_ub)
    ot_test_verses_rnub = remove_proper_nouns(ot_test_verses_no_ub)
    nt_test_verses_rnub = remove_proper_nouns(nt_test_verses_no_ub)

    c_mnb_rnub = BoW_Estimator(MultinomialNB(), " ".join)
    c_mnb_rnub.fit(train_x_removed_nub, train_y)

    (ot_probas_rnub, nt_probas_rnub, ot_mislabels_rnub, nt_mislabels_rnub) = (
        evaluate_classifier(
            c_mnb_rnub,
            ot_test_verses_rnub,
            ot_test_labels,
            nt_test_verses_rnub,
            nt_test_labels,
        )
    )

    save_mislabels(
        ot_mislabels_r,
        nt_mislabels_r,
        formatter=csvify_cal,
        ot_save_file="./out/cal_mnb_prediction_mislabels_ot_removed_both_nub.csv",
        nt_save_file="./out/cal_mnb_prediction_mislabels_nt_removed_both_nub.csv",
    )

    save_all_preds(
        ot_test_verses_r,
        ot_probas_r,
        ot_test_labels,
        nt_test_verses_r,
        nt_probas_r,
        nt_test_labels,
        formatter=csvify_cal,
        ot_save_file="./out/cal_mnb_prediction_all_ot_removed_both_nub.csv",
        nt_save_file="./out/cal_mnb_prediction_all_nt_removed_both_nub.csv",
    )
