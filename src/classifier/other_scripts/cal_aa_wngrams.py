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

"""A script to try some word n-grams."""

from sklearn.naive_bayes import MultinomialNB

from src.classifier.cal_aa_eval import (
    csvify_cal,
    remove_enclitics,
    remove_proper_nouns,
)
from src.classifier.dataset_skeleton import DataSplit
from src.classifier.eval_utils import (
    eval_and_save,
)
from src.classifier.fitting_utils import identity
from src.classifier.load_cal import load_cal_dataset
from src.classifier.wrappers import BoWEstimator

if __name__ == "__main__":
    cal_ds = load_cal_dataset("./")

    train_x = cal_ds.train.get_samples()
    train_y = cal_ds.train.get_labels()

    ot_test_verses = cal_ds.test.get_samples(0)
    nt_test_verses = cal_ds.test.get_samples(1)

    print("\n===================WORD N-GRAMS=========================\n")
    print("\nPlain Classifier")
    print("MultinomialNB")
    c_mnb = BoWEstimator(MultinomialNB(), identity)
    c_mnb.fit(train_x, train_y)

    # train and evaluate
    eval_and_save(
        c_mnb,
        cal_ds,
        csvify_cal,
        out_dir="./classifier/out/",
        save_file_prefix="cal_mnb_word_n_gram",
    )

    print("\nRemove PN & GN")
    print("> Remove PN & GN from the training set")
    print("MultinomialNB")
    # Remove personal names and place names from the training data
    # and train new classifiers
    train_x_removed = remove_proper_nouns(train_x)

    c_mnb_r = BoWEstimator(MultinomialNB(), identity)
    c_mnb_r.fit(train_x_removed, train_y)

    c_mnb_r, probas_pair_r = eval_and_save(
        c_mnb_r,
        cal_ds,
        csvify_cal,
        out_dir="./classifier/out/",
        save_file_prefix="cal_mnb_word_n_gram",
        save_file_suffix="_removed",
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

    c_mnb_rboth = BoWEstimator(MultinomialNB(), identity)
    c_mnb_rboth.fit(train_x_removed, train_y)

    c_mnb_rboth, probas_pair_rboth = eval_and_save(
        c_mnb_rboth,
        cal_ds,
        csvify_cal,
        out_dir="./classifier/out/",
        save_file_prefix="cal_mnb_word_n_gram",
        save_file_suffix="_removed_both",
    )

    print("Remove proclitics")
    print("MultinomialNB")
    w_mnb_noc = BoWEstimator(MultinomialNB(), identity)
    train_x_no_clitics = remove_enclitics(train_x_removed)
    w_mnb_noc.fit(train_x_no_clitics, train_y)

    # train and evaluate
    w_mnb_noc_, probas_noc = eval_and_save(
        w_mnb_noc,
        cal_ds,
        csvify_cal,
        out_dir="./classifier/out/",
        save_file_prefix="cal_mnb_word_n_gram",
        save_file_suffix="_noc",
    )

    print(probas_noc)

    print("> Remove PN, GN, proclitics from the training & test set")
    print("MultinomialNB")
    # Remove personal names and place names from the training data
    # and train new classifiers
    cal_ds.test = DataSplit(
        remove_enclitics(ot_test_verses_r), remove_enclitics(nt_test_verses_r)
    )

    eval_and_save(
        w_mnb_noc,
        cal_ds,
        csvify_cal,
        out_dir="./classifier/out/",
        save_file_prefix="cal_mnb_word_n_gram",
        save_file_suffix="_removed_noc_both",
    )
