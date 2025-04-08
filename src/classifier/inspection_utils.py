"""Utility functions to facilitate inspection of classification results."""
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from classifier.result_utils import (
    Mislabels,
    Verse,
)


def find_mislabels(
        y_correct: list | NDArray,
        y_pred: list | NDArray,
        test_x: list[Verse],
        probas: NDArray
    ) -> Mislabels:
    """Find and return mislabelled verses in string format.

    Args:
        y_correct: list of correct labels
        y_pred: list of predictions
        test_x: list of verses corresponding to the labels
        probas: list of probabilities, of size (numbert of samples,
            number of prediction classes)

    Returns:
        A :class:`Mislabels` instance.
    """
    # initialise storages for mislabelled verse's info
    mislabel_idcs = np.empty(0, dtype=np.int_)
    mislabels = np.empty(0, dtype=np.int_)
    correct_labels = np.empty(0, dtype=np.int_)

    # parse through all the predictions,
    # find and record occasions where they don't match ``y_correct``
    for i in range(len(y_correct)):
        if y_correct[i] != y_pred[i]:
            mislabel_idcs = np.append(mislabel_idcs, i)
            mislabels = np.append(mislabels, y_pred[i])
            correct_labels = np.append(
                    correct_labels,
                    y_correct[i]
                    )

    results = Mislabels(
                            mislabel_idcs,
                            mislabels,
                            correct_labels
                        )

    if len(results) > 0:
        # extract list verses at idcs and set results.verses
        results.extract_verses(test_x)

    if probas is not None:
        results.probas = probas[results.idcs]

    return results


def _sort_arrays(
        objs: NDArray, vals: NDArray
    ) -> tuple[NDArray, NDArray]:
    """Sort the two arrays into the same order.

    Args:
        objs: NDArray holding any objects.
        vals: NDArray with values that can be sorted.

    Returns:
        Two NDArrays sorted in the same way.
        They will be ordered in an ascending manner (small -> big).
    """
    sort_ids = np.argsort(vals)
    sorted_objs = objs[sort_ids]
    sorted_cnts = vals[sort_ids]
    return (sorted_objs, sorted_cnts)


def _sort_counter(cnts: Counter) -> tuple[NDArray, NDArray]:
    """Sort the Counter contents by the counts, return them as np array.

    Args:
        cnts: a :class:`Counter` class object

    Returns:
        Two sorted NDArrays for the Counter keys and values.
        They will be ordered in an ascending manner (small -> big).
    """
    cnt_targets = list(cnts.keys())
    cnt_vals = list(cnts.values())
    return _sort_arrays(np.array(cnt_targets), np.array(cnt_vals))


def find_top_k_words(
    verses: list[Verse],
    top_k: int = 150,
    save_file: str | None = None,
) -> tuple[NDArray | None, NDArray, NDArray]:
    """Find the top k words from the verses.

    Args:
        verses: list of Verse objects
        top_k: number of top k n_grams to discover
        save_file: Name of the file to save.
            Only works when syr_vocabs is provided.

    Returns:
        A tuple of (top k n_grams in Syriac script,
                    top k transliterated n_grams,
                    counts of the top k n_grams.
                    )

    Raises:
        ValueError: if Syriac version is provided but word counts
            do not match up with the transliterated version
    """
    syr_words = []
    translit_words = []

    for vrs in verses:
        translit_words.extend(vrs.get_translit_words())
        if len(vrs.get_syriac_words()) > 0:
            translit_words.extend(vrs.get_syriac_words())

    sorted_translits, sorted_tl_cnts = _sort_counter(Counter(translit_words))

    # revert the order of the sorted arrays
    top_cnts = sorted_tl_cnts[::-1][:top_k]
    top_translits = sorted_translits[::-1][:top_k]

    # validate Syriac script version is available
    if len(syr_words) > 0:
        sorted_syr_words, sorted_syr_cnts = _sort_counter(Counter(syr_words))
        # If Syriac version is provided but word counts do not match up
        # with the transliterated version, raise an exception
        if sorted_syr_cnts.all() != sorted_tl_cnts.all():
            msg = ("Syriac data for words were provided, but the counts of "
                   + "each word from Syriac and Transliterated data "
                   + "do not match!")
            raise ValueError(msg)
        # revert the order of the sorted array
        top_syr_words = sorted_syr_words[::-1][:top_k]
    else:
        top_syr_words = None

    # print(top_cnts[:10])
    # print(top_syr_words[:10])
    # print(top_translits[:10])

    # Save the words to a file
    if save_file is not None and top_syr_words is not None:
        csv_data = "Syriac,Transliteration,Counts"
        for i in range(top_k):
            csv_data += (f"'{top_syr_words[i]}',"
                        f"{top_translits[i]!s},{int(top_cnts[i])}")
    elif save_file is not None:
        # Format the data without Syriac script
        csv_data = "Transliteration,Counts"
        for i in range(top_k):
            csv_data += (f"{top_translits[i]!s},{int(top_cnts[i])}")

        Path(save_file).write_text(csv_data, encoding="utf-8")
    return (top_syr_words, top_translits, top_cnts)


def get_top_n_grams(
        translit_vocabs: Counter,
        syr_vocabs: Counter | None = None,
        top_k: int = 150,
        save_file: str | None = None,
    ) -> tuple[NDArray | None, NDArray, NDArray]:
    """Get top k n-grams, based on the counts stored in translit_vocabs.

    Args:
        translit_vocabs: Counter object for top k vocabs
        syr_vocabs: Couner object for top k Syriac vocabs
        top_k: number of top k n_grams to discover
        save_file: Name of the file to save.
            Only works when syr_vocabs is provided.

    Returns:
        A tuple of (top k n_grams in Syriac script,
                    top k transliterated n_grams,
                    counts of the top k n_grams.
                    )

    Raises:
        ValueError: if cnts of Syriac and transliterated n_grams do not match
    """
    sorted_n_grams, sorted_cnts = _sort_counter(translit_vocabs)

    top_n_grams = sorted_n_grams[::-1][:top_k]
    top_cnts = sorted_cnts[::-1][:top_k]

    # print(top_cnts[:10])
    # print(top_n_grams[:10])

    if syr_vocabs is not None:
        sorted_syr, sorted_syr_cnts = _sort_counter(syr_vocabs)

        if sorted_syr_cnts.all() != sorted_cnts.all():
            msg = (
                "The numbers of counted objects found in"
                + " translit_vocabs and syr_vocabs do not match."
            )
            raise ValueError(msg)

        top_targets_syr = sorted_syr[::-1][:top_k]

        if save_file is not None:
            csv_data = "Syriac,Transliteration,Counts"
            for i in range(top_k):
                csv_data += f"'{''.join(top_targets_syr[i])}',"
                csv_data += f"'{top_n_grams[i]!s}',{int(top_cnts[i])}"
            # print(csv_data.split("")[1])
            Path(save_file).write_text(csv_data, encoding="utf-8")
        return (top_targets_syr, top_n_grams, top_cnts)
    return (None, top_n_grams, top_cnts)
