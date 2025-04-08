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

"""Authorship attribution using data from the ETCBC's dataset."""

import numpy as np
from numpy.typing import NDArray
from sklearn.naive_bayes import MultinomialNB

from classifier.dataset_skeleton import DataSplit
from classifier.eval_utils import (
    eval_and_save,
)
from classifier.fitting_utils import identity
from classifier.result_utils import Verse
from classifier.textfabric_utils import load_etcbc_dataset
from classifier.wrappers import BoWEstimator


def csvify_etcbc(
         samples: list[Verse] | NDArray[Verse],
         probas: list[list[float]] | NDArray[np.float64],
         correct_labels: list[int] | NDArray[np.int64] | None = None
    ) -> str:
    """Convert the ETCBC verse data into a CSV-formatted string.

    Args:
        samples: samples that were labelled.
        probas: two-dimensional list of probabilities of each verse
            belonging to each of the classes.
        correct_labels: gold reference for the samples.

    Returns:
        The ETCBC dataset and prediction data in a string of CSV format.
    """
    if correct_labels is not None:
        # Set header line
        result = (
            '"Reference","Probability for OT","Probability for NT",'
            + '"Correct Label","No. Words","ܐܠܦܒܝܬ ܣܘܪܝܝܐ",'
            + '"ETCBC Transliteration"\n'

        )
        # Extract & format verse data
        for i in range(len(samples)):
            result += (
                f'"{samples[i].reference}",{probas[i][0]:.04f},{probas[i][1]:.04f},'
                + f"{correct_labels[i]}, {len(samples[i])},"
                + f'{" ".join(samples[i].get_syriac_words())},'
                + f'{" ".join(samples[i].get_translit_words())}\n'
            )
    else:
        # Set header line
        result = (
            '"Reference","Probability for OT","Probability for NT",'
            + '"No. Words","ܐܠܦܒܝܬ ܣܘܪܝܝܐ","ETCBC Transliteration"\n'

        )
        # Extract & format verse data
        for i in range(len(samples)):
            result += (
                f'"{samples[i].reference}",{probas[i][0]:.04f},{probas[i][1]:.04f},'
                + f"{len(samples[i])},"
                + f'{" ".join(samples[i].get_syriac_words())},'
                + f'{" ".join(samples[i].get_translit_words())}\n'
            )
    return result


def remove_proper_nouns(verses: list[Verse]) -> list[Verse]:
    """Create a copy of verses with pre-defined proper nouns removed.

    Returns:
        a list of words in each of the verses where the words in the
        ``frequent_propn`` set is excluded.
    """
    frequent_propn = {
        "JCW<",
        "MWC>",
        "J<QWB",
        ">JSRJL",
        "JWSP",
        ">BRHM",
        "CM<WN",
        "LJCW",
    }
    result_verses = []
    for i in range(len(verses)):
        translit_words = verses[i].get_translit_words()
        if len(set(translit_words) & frequent_propn) == 0:
            # if there's no intersection, just add the verse
            result_verses.append(verses[i])
        else:
            # if there is an intersection of verse's words
            # and FREQUENT_PROPER_NOUN
            # print(f"{verses[i].reference} contains a propn!")
            excludes = []
            verse_range = range(len(translit_words))
            for j in verse_range:
                if translit_words[j] in frequent_propn:
                    excludes.append(j)

            translit_r = []
            syriac_r = []
            removed_words = []

            for idx in verse_range:
                if idx not in excludes:
                    translit_r.append(translit_words[idx])
                    syriac_r.append(verses[i].words[idx].syriac)
                else:
                    removed_words.append(verses[i])
            # print(f"Removed {removed} from verses[{i}] "
            # + "({translit_words})")
            result_verses.append(
                    Verse(verses[i].book,
                          verses[i].reference,
                          translit_words=translit_r,
                          syriac_words=syriac_r,
                          origin="ETCBC")
                    )
    return result_verses


if __name__ == "__main__":

    etcbc_ds = load_etcbc_dataset()
    train_verses = etcbc_ds.train.get_samples()
    train_verse_labels = etcbc_ds.train.get_labels()
    ot_test_verses = etcbc_ds.test.get_samples(0)
    nt_test_verses = etcbc_ds.test.get_samples(1)

    # train classifier
    print("\nPlain Classifier")
    mnb = BoWEstimator(MultinomialNB(), " ".join)
    mnb.fit(train_verses, train_verse_labels)

    # evaluate and save results
    eval_and_save(
            mnb,
            etcbc_ds,
            csvify_etcbc,
            out_dir="./classifier/out/",
            save_file_prefix="etcbc_"
            )

    # Remove a few common proper nouns only from the training set
    print("\nRemove common proper nouns from training verses")
    train_verses_removed = remove_proper_nouns(train_verses)
    # assert(train_verses_removed != train_verses)

    mnb_r = BoWEstimator(MultinomialNB(), " ".join)
    mnb_r.fit(train_verses_removed, train_verse_labels)

    # evaluate and save results
    eval_and_save(
            mnb_r,
            etcbc_ds,
            csvify_etcbc,
            out_dir="./classifier/out/",
            save_file_prefix="etcbc_",
            save_file_suffix="_removed"
            )

    # Remove a few common proper nouns from both training and test sets
    print("\nRemove common proper nouns from both training & test verses")
    ot_test_verses_r = remove_proper_nouns(ot_test_verses)
    nt_test_verses_r = remove_proper_nouns(nt_test_verses)
    # assert(train_verses_removed != train_verse_txts)
    etcbc_ds.test = DataSplit(ot_test_verses_r, nt_test_verses_r)

    # mnb_rb = BoWEstimator(MultinomialNB(), " ".join)
    # mnb_rb.fit(train_verses_removed, train_verse_labels)

    # evaluate and save results
    eval_and_save(
            mnb_r,
            etcbc_ds,
            csvify_etcbc,
            out_dir="./classifier/out/",
            save_file_prefix="etcbc_",
            save_file_suffix="_removed_both"
            )

    print("\n===================WORD N-GRAMS=========================\n")
    print("\nPlain Classifier")
    mnb = BoWEstimator(MultinomialNB(), identity)
    mnb.fit(train_verses, train_verse_labels)

    # evaluate and save results
    eval_and_save(
            mnb,
            etcbc_ds,
            csvify_etcbc,
            out_dir="./classifier/out/",
            save_file_prefix="etcbc_"
            )

    # Remove a few common proper nouns only from the training set
    print("\nRemove common proper nouns from training verses")
    train_verses_removed = remove_proper_nouns(train_verses)
    # assert(train_verses_removed != train_verses)

    mnb_r = BoWEstimator(MultinomialNB(), identity)
    mnb_r.fit(train_verses_removed, train_verse_labels)

    # evaluate and save results
    eval_and_save(
            mnb_r,
            etcbc_ds,
            csvify_etcbc,
            out_dir="./classifier/out/",
            save_file_prefix="etcbc_",
            save_file_suffix="_removed"
            )

    # Remove a few common proper nouns from both training and test sets
    print("\nRemove common proper nouns from both training & test verses")
    ot_test_verses_r = remove_proper_nouns(ot_test_verses)
    nt_test_verses_r = remove_proper_nouns(nt_test_verses)
    # assert(train_verses_removed != train_verse_txts)
    etcbc_ds.test = DataSplit(ot_test_verses_r, nt_test_verses_r)

    # mnb_rb = BoWEstimator(MultinomialNB(), " ".join)
    # mnb_rb.fit(train_verses_removed, train_verse_labels)

    # evaluate and save results
    eval_and_save(
            mnb_r,
            etcbc_ds,
            csvify_etcbc,
            out_dir="./classifier/out/",
            save_file_prefix="etcbc_",
            save_file_suffix="_removed_both"
            )
