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
"""Utilities to handle data extraction and estimation results."""
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass
class PeshittaWord:
    """A dataclass to represent a word in the Peshitta texts.

    .. caution::
        This class does not implement __hash__ method, which means that
        this class is unhashable and thus cannot be used as a key in ``dict``
        or as a member of ``set``. This is due to the way __eq__ comparator
        is implemented. Make sure to use one of the attributes as a key when
        using dict.

    Attributes:
        translit: Transliteration of the word. Exact tarnsliteration format
            to be used is dependent on the source dataset. Every instance of
            this class must have this attribute.
        syriac: Original syriac script representation of the word,
            as extracted from the source.
        annots: Annotations on each word.
        origin: Source of the word, where the word data was acquired.
            Use `ETCBC` for `ETCBC/text-fabric <https://github.com/ETCBC/peshitta>`
            and `ETCBC/syrnt <https://github.com/ETCBC/syrnt>`.
            Use `CAL` for the `Comprehensive Aramaic Lexicon <https://cal.huc.edu>`.

    .. note::
        The transliterations and Syriac scripts of the word depends on the
        standards of datasets they were originally acquired from.
    """
    translit: str
    syriac: str = ""
    annots: str = ""
    origin: Literal["ETCBC", "CAL"] = "ETCBC"

    def __eq__(self, value: object, /) -> bool:
        # Since this implementation relies on ``or`` comparison,
        # ``hash((translit, syriac))``, ``hash(translit)`` or ``hash(syriac)``
        # do not satisfy the specification; they may not return the same result
        # even when the result of __eq__ is True.
        try:
            t_translit = value.translit
            t_syriac = value.syriac
        except AttributeError:
            return value in {self.translit, self.syriac}
        # If ``value`` has attributes "translit" and "syriac"
        return (self.translit == t_translit
                or self.syriac == t_syriac)


class Verse(Any):
    """Represents a verse in a biblical text.

    .. note:: A *verse* is a exegetical annotation and is not necessarily
    comparable to semantic divisions like sentence, clause, etc.
    """
    def __init__(  # noqa: PLR0913
            self,
            book_name: str,
            verse_ref: str,
            translit_words: list[str],
            syriac_words: list[str] | None = None,
            words_annotations: list[str] | None = None,
            *,
            origin: Literal["ETCBC", "CAL"] = "ETCBC",
        ) -> None:
        """Initialise Verse object with PeshittaWord attribute.

        Attributes:
            book: Name of the biblical book the verse belongs to.
            reference: A verse reference, e.g. "Genesis Chapter 01 Verse 01"
            words: List of words represented as PeshittaWord objects

        Raises:
            ValueError: if the number of words in transliteration do not match
                with that of Syriac script or the number of word annotations.
        """
        # Initialise the class attrs
        self.book: str = book_name
        self.reference: str = verse_ref
        self.words: list[PeshittaWord] = []

        if (
                syriac_words is not None
                and len(translit_words) != len(syriac_words)
            ):
            msg = ("The number of words in transliteration "
                        + f"{len(translit_words)} and "
                        + f"in Syriac script {len(syriac_words)}"
                        + " do not match.")
            raise ValueError(msg)

        if (
                words_annotations is not None
                and len(translit_words) != len(words_annotations)
            ):
            msg = ("The number of words in transliteration "
                        + f"{len(translit_words)} and "
                        + "the number of annotations "
                        + f"{len(words_annotations)} "
                        + " do not match.")
            raise ValueError(msg)

        # Parse the provided lists and organise them into a PeshittaWord obj
        for i in range(len(translit_words)):
            if syriac_words is not None and words_annotations is not None:
                self.words.append(PeshittaWord(
                        translit_words[i],
                        syriac_words[i],
                        words_annotations[i],
                        origin,
                        ))
            elif syriac_words is not None and words_annotations is None:
                self.words.append(PeshittaWord(
                        translit_words[i],
                        syriac_words[i],
                        origin=origin,
                        ))
            elif syriac_words is None and words_annotations is not None:
                self.words.append(PeshittaWord(
                        translit_words[i],
                        annots=words_annotations[i],
                        origin=origin,
                        ))
            else:
                self.words.append(PeshittaWord(
                        translit_words[i],
                        ))

    def __str__(self) -> str:
        """Return string representation of the verse.

        Returns:
            A string where transliterations words are joined with a space,
            creating a string containing the all words in the verse.
        """
        return " ".join([w.translit for w in self.words])

    def __len__(self) -> int:
        """An under-the-hood method defining the result of :func:`len`.

        Returns:
            An integer indicating the number of words contained in the verse.
        """
        return len(self.words)

    def get_translit_words(self) -> list[str]:
        """Create a transliteration of the verse.

        Similar to ``__str__``, but returns a list of words instead of
        a joined string.

        Returns:
            a list of transliterated words in the verse.

        .. seealso::
            :class:`classifier.result_utils.PeshittaWord`
                Refer to ``translit`` attribute there for details of
                transliterations.
        """
        return [w.translit for w in self.words]

    def get_syriac_words(self) -> list[str]:
        """Create the list of words in Syriac script(s).

        Returns:
            list of words in the verse in Syriac script(s).
            The resulting list may be shorter than that of translit,
            possibly due to some caveats on the source database or
            simply because the Syriac script version was not provided
            upon instantiating the Verse.

        .. seealso::
            :class:`classifier.result_utils.PeshittaWord`
                Refer to ``translit`` attribute there for details of
                Syriac script words.
        """
        return [w.syriac for w in self.words if w.syriac]

    def get_words_in_mode(self, mode: int = 1) -> list[str]:
        """Get the list of words in the script defined by ``mode``.

        Args:
            mode: defines the mode in which to parse the text.
                if ``1``, then use the transliteration.
                if ``2``, then use original Syriac script.

        Returns:
            list of words in the verse in Syriac script(s).

        Raises:
            ValueError: if ``mode`` is not 1 or 2.
        """
        if mode == 1:
            return self.get_translit_words()
        if mode == 2:
            return self.get_syriac_words()
        # else
        msg = (f"Argument `mode` must be 1 or 2, but {mode}"
                    + " was found.")
        raise ValueError(msg)

    def get_annotations(self) -> list[str]:
        """Get ``annots`` attribute of each word in the verse.

        Returns:
            list of annotations for each word in the verse.
        """
        return [w.annots for w in self.words]


class Predictions:
    """A dataclass to hold predictions by some classifier.

    Attributes:
        samples: the original samples provided to the classifier.
        predicted_labels: the labels predicted by the classifier.
        correct_labels: the gold reference labels for each sample.
        num_correct: number of correctly labelled samples.
        num_mislabelled: number of mislabelled samples.
    """

    def __init__(
            self,
            samples: list[Any] | np.ndarray,
            predictions: list[int] | np.ndarray,
            correct_labels: list[int] | np.ndarray,
            num_correct: int | np.int_,
            num_mislabelled: int | np.int_,
        ) -> None:
        self.samples = samples
        self.predictions = predictions
        self.correct_labels = correct_labels
        self.num_correct = num_correct
        self.num_mislabelled = num_mislabelled

        if not (len(self.samples) == len(self.predictions)
                and len(self.samples) == len(self.correct_labels)):
            msg = "length of provided lists/arrays do not match."
            raise ValueError(msg)


class ProbaPredictions(Predictions):
    """A dataclass to hold prediction data with probabilities.

    Attributes:
        probas: probability of each sample belonging to each class, predicted
            by the classifier.
    """
    _probas: list[list[float]] | np.ndarray

    def set_probas(self, probas: list[list[float]] | np.ndarray) -> None:
        """Set the probabilities predicted by the classifier.

        Args:
            probas: list of probabilities, of size (numbert of samples,
                number of prediction classes)
        """
        self._probas = probas

    def get_probas(self) -> list[list[float]] | np.ndarray:
        """Get a list of predicted probabilities.

        Returns:
            list of probabilities, of size (numbert of samples,
                number of prediction classes)

        Raises:
            AttributeError: when this method is called before the attribute
                ``_probas`` is set.
        """
        try:
            return self._probas
        except AttributeError as ae:
            msg = "self._probas is not set. Please run self.set_probas."
            raise AttributeError(msg) from ae

    def save_to_file(self,
                     formatter: Callable,
                     save_file: str | Path = "./out/prediction_all_ot.csv",
                ) -> None:
        """Save all verses into a file, along with prediction results.

        Args:
            formatter: any callable object (function, method, etc.)
                that returns a formatted string which can be directly
                written to a file.
            save_file: file name or path to save the prediction data.

        .. note:: This function uses :meth:`pathlib.Path.write_text` directly.
            Error descriptions in this documentation may be not thorough.
        """
        # save all prediction results to files
        # format the data into strings
        data_str = formatter(samples=self.samples,
                                    probabilities=self.get_probas(),
                                    correct_labels=self.correct_labels
                             )

        # write formatted texts to files
        Path(save_file).write_text(data_str, encoding="utf-8")


class Mislabels:
    """Wrapper of mislabelled results to facilitate human inspection.

    An instance of this class represents a group of mislabelled samples
    from a particular attribution class, or one of the two testaments.

    Attributes:
        idcs: indices of mislabelled samples
        mislabels: incorrect labels the classifier assigned
        correct_labels: gold references for the verses
        verses: mislabelled verses corresponding to idcs
        probas: probabilities of the mislabelled verses
    """
    def __init__(
            self,
            indices: list | np.ndarray,
            incorrect_labels: list | np.ndarray,
            correct_labels: list | np.ndarray,
            probas: list | np.ndarray | None = None
        ) -> None:
        self.idcs: np.ndarray = np.array(indices, dtype="int_")

        if len(self.idcs) < 1:
            msg = ("indices of mislabelled samples were empty."
                   + "Make sure to instanciate this only when "
                   + "there are one or more mislabelled samples.")
            raise ValueError(msg)

        self.mislabels: np.ndarray = np.array(incorrect_labels, dtype=int)
        self.correct_labels: np.ndarray = np.array(correct_labels, dtype=int)
        # initialise empty variables
        self.verses = []
        if probas is not None:
            self.probas = np.array(probas)

    def __len__(self) -> int:
        """An under-the-hood method defining the result of :func:`len`.

        Returns:
            An integer indicating number of mislabelled verses.
        """
        return len(self.idcs)

    def extract_verses(
            self,
            vrs: list[Verse],
        ) -> None:
        """Set self.verses by extracting verses at self.idcs.

        Args:
            vrs: a list of Verses which includes some mislabelled verses.
                The order of verses must correspond to those of self.idcs.
        """
        self.verses = [vrs[idx] for idx in self.idcs]

    def save_to_file(
            self,
            formatter: Callable,
            save_file: str | Path = "./out/prediction_mislabels.csv",
        ) -> None:
        """Save mislabelled verses into a file.

        Args:
            formatter: any :class:`Callable` object (function, method, etc.)
                that returns a formatted string which can be directly
                written to a file. The callable must accept arguments
                ``X``, ``probas``, and ``y_correct``.
            save_file: :class:`pathlib.Path` obj or string containing
                a path to save the mislabelled verses' data.
        """
        save_data = formatter(
            samples=self.verses,
            probas=self.probas,
            correct_labels=self.correct_labels
        )

        # write formatted texts to files
        Path(save_file).write_text(save_data, encoding="utf-8")
