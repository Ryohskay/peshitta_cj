# BSD 2-Clause License
#
# Copyright (c) 2025, Ryosuke Nagata
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
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

"""Utility functions to facilitate inspection of classification results."""
from collections import Counter
from pathlib import Path

from classifier.result_utils import (
    Verse,
)


def find_top_k_words(
        verses: list[Verse],
        top_k: int = 150,
        save_file: str | None = None,
    ) -> tuple[list[tuple[str, int]], list[tuple[str, int]] | None]:
    """Find the top k words from the verses.

    Args:
        verses: list of Verse objects
        top_k: number of top k n_grams to discover
        save_file: Name of the file to save.
            Only works when syr_vocabs is provided.

    Returns:
        A tuple of (
            list of tuples containing transliterated word and its count
            list of tuples containing syriac word and its count
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

    sorted_translits = Counter(translit_words).most_common(top_k)

    # validate Syriac script version is available
    if len(syr_words) > 0:
        sorted_syrs = Counter(syr_words).most_common(top_k)
        # If Syriac version is provided but word counts do not match up
        # with the transliterated version, raise an exception
        if [w[1] for w in sorted_syrs] != [w[1] for w in sorted_translits]:
            msg = ("Syriac data for words were provided, but the counts of "
                   + f"each word from Syriac {len(sorted_syrs)} and "
                   + "Transliterated data {len(sorted_translits)} "
                   + "do not match!")
            raise ValueError(msg)
    else:
        sorted_syrs = None

    # Save the words to a file
    if save_file is not None and sorted_syrs is not None:
        csv_data = "Syriac,Transliteration,Counts"
        for i in range(top_k):
            csv_data += (f"'{sorted_syrs[i][0]}',"
                        + f"{sorted_translits[i][0]},"
                        + f"{int(sorted_translits[i][1])}")
    elif save_file is not None:
        # Format the data without Syriac script
        csv_data = "Transliteration,Counts"
        for i in range(top_k):
            csv_data += (f"{sorted_translits[i][0]},"
                         + f"{int(sorted_translits[i][1])}")

        Path(save_file).write_text(csv_data, encoding="utf-8")
    return (sorted_translits, sorted_syrs)


def get_top_n_grams(
        translit_vocabs: Counter,
        syr_vocabs: Counter | None = None,
        top_k: int = 150,
        save_file: str | None = None,
    ) -> tuple[list[tuple[str, int]], list[tuple[str, int]] | None]:
    """Get top k n-grams, based on the counts stored in translit_vocabs.

    Args:
        translit_vocabs: Counter object for top k vocabs
        syr_vocabs: Couner object for top k Syriac vocabs
        top_k: number of top k n_grams to discover
        save_file: Name of the file to save.
            Only works when syr_vocabs is provided.

    Returns:
        A tuple of (
            list of tuples containing transliterated char n-grams and its count
            list of tuples containing syriac char n-grams and its count
            )

    Raises:
        ValueError: if cnts of Syriac and transliterated n_grams do not match
    """
    top_n_grams = translit_vocabs.most_common(top_k)

    if syr_vocabs is not None:
        top_syr_vocabs = syr_vocabs.most_common(top_k)

        if top_n_grams != top_syr_vocabs:
            msg = (
                "The numbers of counted objects found in"
                + " translit_vocabs and syr_vocabs do not match."
            )
            raise ValueError(msg)

        if save_file is not None:
            csv_data = "Syriac,Transliteration,Counts"
            for i in range(top_k):
                csv_data += f"'{''.join(top_syr_vocabs[i][0])}',"
                csv_data += f"'{''.join(top_n_grams[i][0])}',{int(top_n_grams[i][1])}"
            # print(csv_data.split("")[1])
            Path(save_file).write_text(csv_data, encoding="utf-8")
    else:
        top_syr_vocabs = None
    return (top_n_grams, top_syr_vocabs)
