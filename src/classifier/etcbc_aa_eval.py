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

from src.classifier.dataset_skeleton import DataSplit
from src.classifier.eval_utils import (
    eval_and_save,
)
from src.classifier.fitting_utils import identity
from src.classifier.result_utils import Verse
from src.classifier.textfabric_utils import load_etcbc_dataset
from src.classifier.wrappers import BoWEstimator
from src.shared import label_data
from src.classifier.fname_utils import SavefileName, FnameExtraOpts


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
            '"Book","Reference",'
            + f'"Probability for {label_data.ValToLabel[0]}",'
            + f'"Probability for {label_data.ValToLabel[1]}",'
            + '"ETCBC Transliteration","ܐܠܦܒܝܬ ܣܘܪܝܝܐ",'
            + '"Correct Label"\n'
        )
        # Extract & format verse data
        for i in range(len(samples)):
            result += (
                f'"{samples[i].book}","{samples[i].reference}",{probas[i][0]:.04f},{probas[i][1]:.04f},'
                + f"{' '.join(samples[i].get_translit_words())},"
                + f"{' '.join(samples[i].get_syriac_words())},"
                + f"{correct_labels[i]}\n"
            )
    else:
        # Set header line
        result = (
            '"Book","Reference",'
            + f'"Probability for {label_data.ValToLabel[0]}",'
            + f'"Probability for {label_data.ValToLabel[1]}",'
            + '"ETCBC Transliteration","ܐܠܦܒܝܬ ܣܘܪܝܝܐ"\n'
        )
        # Extract & format verse data
        for i in range(len(samples)):
            result += (
                f'"{samples[i].book}","{samples[i].reference}",{probas[i][0]:.04f},{probas[i][1]:.04f},'
                + f'{" ".join(samples[i].get_translit_words())},'
                + f'{" ".join(samples[i].get_syriac_words())}\n'
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


def _replace_diacritics(s: str) -> str:
    # replace Syriac diacritics
    s = s.replace("\u0308", "").replace("\u0307", "")
    # replace transliteration of above diacritics
    return s.replace('"', "").replace("^", "")


def remove_non_chars(verses: list[list[str]]) -> list[list[str]]:
    """Remove non-character unicode codepoints from the text.

    Specifically, this preprocessing function removes ``\u0308``
    (Combining Diaeresis) used in place of Syriac diacritic Seyame (ܣܝ̈ܡܐ)
    and ``\u0307`` (Combining Dot Above).

    Returns:
        a list of str without non-character unicode codepoints.
    """
    new_verses = []
    for v in verses:
        new_verses.append([_replace_diacritics(word) for word in v])
    return new_verses


if __name__ == "__main__":

    etcbc_ds = load_etcbc_dataset()
    train_verses = remove_non_chars(etcbc_ds.train.get_samples())
    train_verse_labels = etcbc_ds.train.get_labels()
    ot_test_verses = remove_non_chars(etcbc_ds.test.get_samples(0))
    nt_test_verses = remove_non_chars(etcbc_ds.test.get_samples(1))

    # train classifier
    print("\nPlain Classifier")
    mnb = BoWEstimator(MultinomialNB(), " ".join)
    mnb.fit(train_verses, train_verse_labels)

    # evaluate and save results
    save_fname = SavefileName("ETCBC", "mnb")
    save_fname.set_ngram_opts()
    eval_and_save(
            mnb,
            etcbc_ds,
            csvify_etcbc,
            save_fname,
            out_dir="./src/classifier/out/"
            )

    # Remove a few common proper nouns only from the training set
    print("\nRemove common proper nouns from training verses")
    train_verses_removed = remove_proper_nouns(train_verses)
    # assert(train_verses_removed != train_verses)

    mnb_r = BoWEstimator(MultinomialNB(), " ".join)
    mnb_r.fit(train_verses_removed, train_verse_labels)

    # evaluate and save results
    save_fname_r = save_fname.copy()
    save_fname_r.add_extra_opts([FnameExtraOpts.REMOVE_PROPN])
    eval_and_save(
            mnb_r,
            etcbc_ds,
            csvify_etcbc,
            save_fname_r,
            out_dir="./src/classifier/out/"
            )

    # Remove a few common proper nouns from both training and test sets
    print("\nRemove common proper nouns from both training & test verses")
    ot_test_verses_r = remove_proper_nouns(ot_test_verses)
    nt_test_verses_r = remove_proper_nouns(nt_test_verses)
    # assert(train_verses_removed != train_verse_txts)
    etcbc_ds.test = DataSplit(ot_test_verses_r, nt_test_verses_r)

    # evaluate and save results
    save_fname_rb = save_fname.copy()
    save_fname_rb.add_extra_opts([FnameExtraOpts.REMOVE_PROPN,
                                    FnameExtraOpts.REMOVE_FROM_BOTH])
    eval_and_save(
            mnb_r,
            etcbc_ds,
            csvify_etcbc,
            save_fname_rb,
            out_dir="./src/classifier/out/"
            )

    print("\n===================WORD N-GRAMS=========================\n")
    print("\nPlain Classifier")
    mnb_w = BoWEstimator(MultinomialNB(), identity)
    mnb_w.fit(train_verses, train_verse_labels)

    # evaluate and save results
    save_fname_w = save_fname.copy()
    save_fname_w.set_ngram_opts(is_char_level=False)
    eval_and_save(
            mnb_w,
            etcbc_ds,
            csvify_etcbc,
            save_fname_w,
            out_dir="./src/classifier/out/",
            )

    # Remove a few common proper nouns only from the training set
    print("\nRemove common proper nouns from training verses")
    train_verses_removed = remove_proper_nouns(train_verses)
    # assert(train_verses_removed != train_verses)

    mnb_wr = BoWEstimator(MultinomialNB(), identity)
    mnb_wr.fit(train_verses_removed, train_verse_labels)

    # evaluate and save results
    save_fname_wr = save_fname_w.copy()
    save_fname_wr.add_extra_opts([FnameExtraOpts.REMOVE_PROPN])
    eval_and_save(
            mnb_wr,
            etcbc_ds,
            csvify_etcbc,
            save_fname_wr,
            out_dir="./src/classifier/out/"
            )

    # Remove a few common proper nouns from both training and test sets
    print("\nRemove common proper nouns from both training & test verses")
    ot_test_verses_r = remove_proper_nouns(ot_test_verses)
    nt_test_verses_r = remove_proper_nouns(nt_test_verses)
    # assert(train_verses_removed != train_verse_txts)
    etcbc_ds.test = DataSplit(ot_test_verses_r, nt_test_verses_r)

    # evaluate and save results
    save_fname_wrb = save_fname_w.copy()
    save_fname_wrb.extra_opts([FnameExtraOpts.REMOVE_PROPN,
                                FnameExtraOpts.REMOVE_FROM_BOTH])
    eval_and_save(
            mnb_wr,
            etcbc_ds,
            csvify_etcbc,
            save_fname_wrb,
            out_dir="./src/classifier/out/",
            )
