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

import numpy as np
from numpy.typing import NDArray
from sklearn.naive_bayes import MultinomialNB

from classifier.cal_aa_eval import csvify_cal, remove_proclitic_ubs, remove_proper_nouns
from classifier.dataset_skeleton import DataSplit
from classifier.eval_utils import (
    cross_validate,
    eval_and_save,
)
from classifier.fitting_utils import identity
from classifier.load_cal import load_cal_dataset
from classifier.result_utils import Verse
from classifier.wrappers import BoWEstimator

if __name__ == "__main__":
    cal_ds = load_cal_dataset("./")

    for n in range(1,6):
        print(f"\n +++++++++++++++++++++++++++++ N={n} ++++++++++++++++++++++++++++")

        train_x = cal_ds.train.get_samples()
        train_y = cal_ds.train.get_labels()

        print("\nPlain Classifier")
        print("MultinomialNB")
        c_mnb = BoWEstimator(MultinomialNB(), " ".join, n)
        cross_validate(c_mnb, train_x, train_y)
        c_mnb.fit(train_x, train_y)

        # train and evaluate
        eval_and_save(
                    c_mnb,
                    cal_ds,
                    csvify_cal,
                    out_dir="./classifier/out/",
                    save_file_prefix=f"cal_mnb_char_{n}gram_",
                )

        print("\nRemove underscores marking proclitics, from training set")
        print("MultinomialNB")
        train_x_no_ub = remove_proclitic_ubs(train_x)

        c_mnb_nub = BoWEstimator(MultinomialNB(), " ".join, n)
        cross_validate(c_mnb_nub, train_x_no_ub, train_y)
        c_mnb_nub.fit(train_x_no_ub, train_y)

        c_mnb_nub, probas_pair_nub = eval_and_save(
                                    c_mnb_nub,
                                    cal_ds,
                                    csvify_cal,
                                    out_dir="./classifier/out/",
                                    save_file_prefix=f"cal_mnb_char_{n}gram_",
                                    save_file_suffix="_nub"
                                )

        print("\nRemove PN & GN")
        print("> Remove PN & GN from the training set")
        print("MultinomialNB")
        # Remove personal names and place names from the training data
        # and train new classifiers
        train_x_removed = remove_proper_nouns(train_x)

        c_mnb_r = BoWEstimator(MultinomialNB(), " ".join, n)
        cross_validate(c_mnb_r, train_x_removed, train_y)
        c_mnb_r.fit(train_x_removed, train_y)

        c_mnb_r, probas_pair_r = eval_and_save(
                                    c_mnb_r,
                                    cal_ds,
                                    csvify_cal,
                                    out_dir="./classifier/out/",
                                    save_file_prefix=f"cal_mnb_char_{n}gram_",
                                    save_file_suffix="_removed"
                                )

        print("\nRemove PN, GN, underscores")
        print("> Remove PN, GN, underscores from training set")
        print("MultinomialNB")
        # Remove personal names and place names from
        # both the training and test datasets
        # and train new classifiers
        train_x_removed_nub = remove_proper_nouns(train_x_no_ub)

        c_mnb_rnub = BoWEstimator(MultinomialNB(), " ".join, n)
        cross_validate(c_mnb_rnub, train_x_removed_nub, train_y)
        c_mnb_rnub.fit(train_x_removed_nub, train_y)

        c_mnb_rnub, probas_pair_rnub = eval_and_save(
                                    c_mnb_rnub,
                                    cal_ds,
                                    csvify_cal,
                                    out_dir="./classifier/out/",
                                    save_file_prefix=f"cal_mnb_char_{n}gram_",
                                    save_file_suffix="_removed_nub"
                                )

        print("\n!!!!!!!!!!!!!!!BELOW REQUIRES DS-wide processing!!!!!!!!!!!!!!!")
        ot_test_verses = cal_ds.test.get_samples(0)
        nt_test_verses = cal_ds.test.get_samples(1)
        print("\nRemove underscores marking proclitics, "
                + "from both training & test sets")
        print("MultinomialNB")
        ot_test_verses_no_ub = remove_proclitic_ubs(ot_test_verses)
        nt_test_verses_no_ub = remove_proclitic_ubs(nt_test_verses)

        cal_ds.test = DataSplit(ot_test_verses_no_ub, nt_test_verses_no_ub)

        c_mnb_nub_both = BoWEstimator(MultinomialNB(), " ".join, n)
        c_mnb_nub_both.fit(train_x_no_ub, train_y)

        c_mnb_nub_both, probas_pair_nub_both = eval_and_save(
                                    c_mnb_nub_both,
                                    cal_ds,
                                    csvify_cal,
                                    out_dir="./classifier/out/",
                                    save_file_prefix=f"cal_mnb_char_{n}gram_",
                                    save_file_suffix="_nub_both"
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

        c_mnb_rboth = BoWEstimator(MultinomialNB(), " ".join, n)
        c_mnb_rboth.fit(train_x_removed, train_y)

        c_mnb_rboth, probas_pair_rboth = eval_and_save(
                                    c_mnb_rboth,
                                    cal_ds,
                                    csvify_cal,
                                    out_dir="./classifier/out/",
                                    save_file_prefix=f"cal_mnb_char_{n}gram_",
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

        c_mnb_rboth_nub = BoWEstimator(MultinomialNB(), " ".join, n)
        c_mnb_rboth_nub.fit(train_x_removed_both_nub, train_y)

        c_mnb_rboth_nub, probas_pair_rboth_nub = eval_and_save(
                                    c_mnb_rboth_nub,
                                    cal_ds,
                                    csvify_cal,
                                    out_dir="./classifier/out/",
                                    save_file_prefix=f"cal_mnb_char_{n}gram_",
                                    save_file_suffix="_removed_both_nub"
                                )

        print("\n===================WORD N-GRAMS=========================\n")
        print("\nPlain Classifier")
        print("MultinomialNB")
        w_mnb = BoWEstimator(MultinomialNB(), identity, n)
        cross_validate(w_mnb, train_x, train_y)
        w_mnb.fit(train_x, train_y)

        # train and evaluate
        eval_and_save(
                    w_mnb,
                    cal_ds,
                    csvify_cal,
                    out_dir="./classifier/out/",
                    save_file_prefix=f"cal_mnb_word_{n}gram_",
                )

        print("\nRemove PN & GN")
        print("> Remove PN & GN from the training set")
        print("MultinomialNB")
        # Remove personal names and place names from the training data
        # and train new classifiers
        train_x_removed = remove_proper_nouns(train_x)

        w_mnb_r = BoWEstimator(MultinomialNB(), identity, n)
        cross_validate(w_mnb, train_x_removed, train_y)
        w_mnb_r.fit(train_x_removed, train_y)

        w_mnb_r, probas_pair_r = eval_and_save(
                                    w_mnb_r,
                                    cal_ds,
                                    csvify_cal,
                                    out_dir="./classifier/out/",
                                    save_file_prefix=f"cal_mnb_word_{n}gram_",
                                    save_file_suffix="_removed"
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

        c_mnb_rboth = BoWEstimator(MultinomialNB(), identity, n)
        c_mnb_rboth.fit(train_x_removed, train_y)

        c_mnb_rboth, probas_pair_rboth = eval_and_save(
                                    c_mnb_rboth,
                                    cal_ds,
                                    csvify_cal,
                                    out_dir="./classifier/out/",
                                    save_file_prefix=f"cal_mnb_word_{n}gram_",
                                    save_file_suffix="_removed_both"
                                )
