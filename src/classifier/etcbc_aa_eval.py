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

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.naive_bayes import MultinomialNB

from src.classifier.dataset_skeleton import DataSplit, LoadedDataset
from src.classifier.eval_utils import (
    eval_and_save,
)
from src.classifier.fitting_utils import identity
from src.classifier.fname_utils import FnameExtraOpts, SavefileName
from src.classifier.result_utils import Verse
from src.classifier.textfabric_utils import load_etcbc_dataset
from src.classifier.wrappers import BoWEstimator
from src.shared import label_data


def csvify_etcbc(
    samples: list[Verse] | NDArray[Verse],
    probas: list[list[float]] | NDArray[np.float64],
    correct_labels: list[int] | NDArray[np.int64] | None = None,
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
                + f"{' '.join(samples[i].get_translit_words())},"
                + f"{' '.join(samples[i].get_syriac_words())}\n"
            )
    return result


def remove_proper_nouns(verse: Verse) -> Verse:
    """Remove pre-defined proper nouns from the verse.

    Returns:
        a :class:`src.classifier.result_utils.Verse` instance where the words
        in the set ``_frequent_propn`` is excluded.
    """
    _frequent_propn = {
        "JCW<",
        "MWC>",
        "J<QWB",
        ">JSRJL",
        "JWSP",
        ">BRHM",
        "CM<WN",
        "LJCW",
    }
    translit_words = verse.get_translit_words()
    if len(set(translit_words) & _frequent_propn) == 0:
        # if there's no intersection, just return the verse as is
        return verse
    # elif there is an intersection of the verse's words
    # and FREQUENT_PROPER_NOUN
    # print(f"{verse.reference} contains a propn!")
    excludes = []
    verse_range = range(len(translit_words))
    for j in verse_range:
        if translit_words[j] in _frequent_propn:
            excludes.append(j)

    translit_r = []
    syriac_r = []
    removed_words = []

    for idx in verse_range:
        if idx not in excludes:
            translit_r.append(translit_words[idx])
            syriac_r.append(verse.words[idx].syriac)
        else:
            removed_words.append(verse)
    # print(f"Removed {removed_words} from ({verse.reference})"
    # + "({translit_words})")
    return Verse(
        verse.book,
        verse.reference,
        translit_words=translit_r,
        syriac_words=syriac_r,
        origin="ETCBC",
    )


def remove_non_chars(verse: list[str]) -> list[str]:
    """Remove non-character unicode codepoints from the text.

    Specifically, this preprocessing function removes ``\u0308``
    (Combining Diaeresis) used in place of Syriac diacritic Seyame (ܣܝ̈ܡܐ)
    and ``\u0307`` (Combining Dot Above).

    Returns:
        a list of str without non-character unicode codepoints.
    """
    res_verse = []
    for i in range(len(verse)):
        # replace Syriac diacritics
        word = verse[i].replace("\u0308", "").replace("\u0307", "")
        # replace transliterations of above diacritics
        res_verse.append(word.replace('"', "").replace("^", ""))
    return res_verse


def etcbc_eval_classifier(
    clf: BoWEstimator,
    etcbc_load: LoadedDataset,
    save_f: SavefileName,
    func_to_map: Callable[[Verse], Verse | None] | None = None,
    *,
    map_to_both: bool = False,
) -> None:
    """Evaluate a classifier with the ETCBC data."""
    train_x = etcbc_load.train.get_samples()
    train_y = etcbc_load.train.get_labels()

    if func_to_map is not None:
        train_x = list(map(func_to_map, train_x))
        if map_to_both:
            test_x_ot = list(map(func_to_map, etcbc_load.test.get_samples(0)))
            test_x_nt = list(map(func_to_map, etcbc_load.test.get_samples(1)))
            etcbc_load.test = DataSplit(test_x_ot, test_x_nt)

    clf.fit(train_x, train_y)

    eval_and_save(
        clf,
        etcbc_load,
        csvify_etcbc,
        save_f,
        out_dir="./src/classifier/out/",
    )


if __name__ == "__main__":
    etcbc_ds = load_etcbc_dataset()
    n_window = 3

    print("\n==================ERRONEOUS CHAR UNI-GRAM================")
    print("\nChar Uni-gram Classifier WITHOUT removing non chars")
    # erroneous results but interesting example of data leakage
    c = BoWEstimator(MultinomialNB(), " ".join, n=1)
    c_fname = SavefileName("ETCBC", "mnb")
    c_fname.set_ngram_opts(n=1)
    c_fname.add_extra_opts([FnameExtraOpts.IS_ERRONEOUS])
    etcbc_eval_classifier(c, etcbc_ds, c_fname)

    print("\nChar Uni-gram Classifier AFTER removing non chars")
    # correct impl of char uni-gram classifier
    c = BoWEstimator(MultinomialNB(), " ".join, n=1)
    c.set_preprocessor(remove_non_chars)
    c_fname = SavefileName("ETCBC", "mnb")
    c_fname.set_ngram_opts(n=1)
    c_fname.add_extra_opts([FnameExtraOpts.REMOVE_DIACRITICS])
    etcbc_eval_classifier(c, etcbc_ds, c_fname)

    print("\n==================REAL RESULTS================")
    print("\nPlain Classifier")
    mnb = BoWEstimator(MultinomialNB(), " ".join, n=n_window)
    mnb.set_preprocessor(remove_non_chars)

    # evaluate and save results
    save_fname = SavefileName("ETCBC", "mnb")
    save_fname.set_ngram_opts(n=n_window)
    save_fname.add_extra_opts([FnameExtraOpts.REMOVE_DIACRITICS])
    etcbc_eval_classifier(mnb, etcbc_ds, save_fname)

    # Remove a few common proper nouns only from the training set
    print("\nRemove common proper nouns from training verses")
    # assert(train_verses_removed != train_verses)
    mnb_r = BoWEstimator(MultinomialNB(), " ".join, n=n_window)
    mnb_r.set_preprocessor(remove_non_chars)

    # evaluate and save results
    save_fname_r = save_fname.copy()
    save_fname_r.add_extra_opts([FnameExtraOpts.REMOVE_PROPN])
    etcbc_eval_classifier(mnb_r, etcbc_ds, save_fname_r, remove_proper_nouns)

    # Remove a few common proper nouns from both training and test sets
    print("\nRemove common proper nouns from both training & test verses")
    # evaluate and save results
    save_fname_rb = save_fname.copy()
    save_fname_rb.add_extra_opts(
        [FnameExtraOpts.REMOVE_PROPN, FnameExtraOpts.REMOVE_FROM_BOTH]
    )
    etcbc_eval_classifier(
        mnb_r, etcbc_ds, save_fname_rb, remove_proper_nouns, map_to_both=True
    )

    print("\n===================WORD N-GRAMS=========================")
    print("\nPlain Classifier")
    mnb_w = BoWEstimator(MultinomialNB(), identity, n_window)
    mnb_w.set_preprocessor(remove_non_chars)
    # evaluate and save results
    save_fname_w = save_fname.copy()
    save_fname_w.set_ngram_opts(n=n_window, is_char_level=False)
    etcbc_eval_classifier(mnb_w, etcbc_ds, save_fname_w)

    # Remove a few common proper nouns only from the training set
    print("\nRemove common proper nouns from training verses")
    # configure classifier
    mnb_wr = BoWEstimator(MultinomialNB(), identity, n=n_window)
    mnb_wr.set_preprocessor(remove_non_chars)
    # evaluate and save results
    save_fname_wr = save_fname_w.copy()
    save_fname_wr.add_extra_opts([FnameExtraOpts.REMOVE_PROPN])
    etcbc_eval_classifier(mnb_wr, etcbc_ds, save_fname_wr, remove_proper_nouns)

    # Remove a few common proper nouns from both training and test sets
    print("\nRemove common proper nouns from both training & test verses")
    # evaluate and save results
    save_fname_wrb = save_fname_w.copy()
    save_fname_wrb.add_extra_opts(
        [FnameExtraOpts.REMOVE_PROPN, FnameExtraOpts.REMOVE_FROM_BOTH]
    )
    etcbc_eval_classifier(
        mnb_wr, etcbc_ds, save_fname_wrb, remove_proper_nouns, map_to_both=True
    )
