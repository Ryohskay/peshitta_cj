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

from sklearn.naive_bayes import MultinomialNB

from classifier.dataset_skeleton import DataSplit
from classifier.etcbc_aa_eval import csvify_etcbc, remove_proper_nouns
from classifier.eval_utils import (
    cross_validate,
    eval_and_save,
)
from classifier.fitting_utils import identity
from classifier.result_utils import Verse
from classifier.textfabric_utils import load_etcbc_dataset
from classifier.wrappers import BoWEstimator


def remove_non_chars(verses: list[Verse]) -> list[Verse]:
    new_verses = []
    for v in verses:
        syriac = [sw.replace("\u0308", "").replace("\u0307", "") for sw in v.get_syriac_words()]
        translit = [tw.replace('"', "").replace("^", "") for tw in v.get_translit_words()]
        new_verses.append(Verse(v.book, v.reference, translit, syriac,
                                v.get_annotations(), origin="ETCBC"))
    return new_verses


if __name__ == "__main__":
    etcbc_ds = load_etcbc_dataset()

    for n in range(1, 6):
        print(f"\n +++++++++++++++++++++++++++++ N={n} ++++++++++++++++++++++++++++")

        train_x = remove_non_chars(etcbc_ds.train.get_samples())
        etcbc_ds.train.verses = remove_non_chars(etcbc_ds.train.get_samples())
        train_y = etcbc_ds.train.get_labels()

        print("\nPlain Classifier")
        print("MultinomialNB")
        c_mnb = BoWEstimator(MultinomialNB(), " ".join, n)
        cross_validate(c_mnb, train_x, train_y)
        c_mnb.fit(train_x, train_y)

        # train and evaluate
        eval_and_save(
                    c_mnb,
                    etcbc_ds,
                    csvify_etcbc,
                    out_dir="./classifier/out/",
                    save_file_prefix=f"etcbc_mnb_char_{n}gram_",
                )

        print("\nRemove Proper nouns")
        print("> Remove common proper nouns from the training set")
        print("MultinomialNB")
        # Remove personal names and place names from the training data
        # and train new classifiers
        train_x_removed = remove_proper_nouns(train_x)

        c_mnb_r = BoWEstimator(MultinomialNB(), " ".join, n)
        cross_validate(c_mnb_r, train_x_removed, train_y)
        c_mnb_r.fit(train_x_removed, train_y)

        c_mnb_r, probas_pair_r = eval_and_save(
                                    c_mnb_r,
                                    etcbc_ds,
                                    csvify_etcbc,
                                    out_dir="./classifier/out/",
                                    save_file_prefix=f"etcbc_mnb_char_{n}gram_",
                                    save_file_suffix="_removed"
                                )

        print("\n!!!!!!!!!!!!!!!BELOW REQUIRES DS-wide processing!!!!!!!!!!!!!!!")
        ot_test_verses = etcbc_ds.test.get_samples(0)
        nt_test_verses = etcbc_ds.test.get_samples(1)

        print("\nRemove Proper nouns")
        print("> Remove common proper nouns from both training & test sets")
        print("MultinomialNB")
        # Remove personal names and place names from
        # both the training and test datasets
        # and train new classifiers
        ot_test_verses_r = remove_proper_nouns(ot_test_verses)
        nt_test_verses_r = remove_proper_nouns(nt_test_verses)

        etcbc_ds.test = DataSplit(ot_test_verses_r, nt_test_verses_r)

        c_mnb_rboth = BoWEstimator(MultinomialNB(), " ".join, n)
        c_mnb_rboth.fit(train_x_removed, train_y)

        c_mnb_rboth, probas_pair_rboth = eval_and_save(
                                    c_mnb_rboth,
                                    etcbc_ds,
                                    csvify_etcbc,
                                    out_dir="./classifier/out/",
                                    save_file_prefix=f"etcbc_mnb_char_{n}gram_",
                                    save_file_suffix="_removed_both"
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
                    etcbc_ds,
                    csvify_etcbc,
                    out_dir="./classifier/out/",
                    save_file_prefix=f"etcbc_mnb_word_{n}gram_",
                )

        print("\nRemove Proper nouns")
        print("> Remove common proper nouns from the training set")
        print("MultinomialNB")
        # Remove personal names and place names from the training data
        # and train new classifiers
        train_x_removed = remove_proper_nouns(train_x)

        w_mnb_r = BoWEstimator(MultinomialNB(), identity, n)
        cross_validate(w_mnb, train_x_removed, train_y)
        w_mnb_r.fit(train_x_removed, train_y)

        w_mnb_r, probas_pair_r = eval_and_save(
                                    w_mnb_r,
                                    etcbc_ds,
                                    csvify_etcbc,
                                    out_dir="./classifier/out/",
                                    save_file_prefix=f"etcbc_mnb_word_{n}gram_",
                                    save_file_suffix="_removed"
                                )

        print("\nRemove Proper nouns")
        print("> Remove common proper nouns from both training & test sets")
        print("MultinomialNB")
        # Remove personal names and place names from
        # both the training and test datasets
        # and train new classifiers
        ot_test_verses_r = remove_proper_nouns(ot_test_verses)
        nt_test_verses_r = remove_proper_nouns(nt_test_verses)

        etcbc_ds.test = DataSplit(ot_test_verses_r, nt_test_verses_r)

        c_mnb_rboth = BoWEstimator(MultinomialNB(), identity, n)
        c_mnb_rboth.fit(train_x_removed, train_y)

        c_mnb_rboth, probas_pair_rboth = eval_and_save(
                                    c_mnb_rboth,
                                    etcbc_ds,
                                    csvify_etcbc,
                                    out_dir="./classifier/out/",
                                    save_file_prefix=f"etcbc_mnb_word_{n}gram_",
                                    save_file_suffix="_removed_both"
                                )
