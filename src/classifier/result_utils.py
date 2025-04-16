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

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from src.classifier.sanitisation_utils import clean_path_str

logger = logging.getLogger(__name__)


def append_to_dict(key: str, values: list, target: dict[str, list]) -> dict:
    """A utility function to append a list to a dictionary.

    If a ``key`` doesn't yet exist in the ``target`` dictionary, then update
    the dictionary as usual.

    If ``key`` already exists, instead of replacing the value, append the list
    at the end of the existing list value.

    Returns:
        an updated dict with the given values appended.
    """
    if key not in target:
        target.update({key: values})
    else:
        target[key].append(values)
    return target


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
        """Equality comparator for Verse objects with any Python object.

        Returns:
            A Bernoulli (boolean) value indicating if ``self`` and ``value``
            are considered *equal*.

        .. attention:: Since this implementation relies on ``or`` comparison,
            ``hash((translit, syriac))``, ``hash(translit)`` or ``hash(syriac)``
            do not satisfy the specification; they may not return the same
            result even when the result of __eq__ is True.
        """
        t_translit = ""
        t_syriac = ""
        if (hasattr(value, "translit") and hasattr(value, "syriac")):
            # If ``value`` has attributes "translit" and "syriac"
            t_translit = value.translit  # type: ignore[reportAttributeAccessIssue]
            t_syriac = value.syriac  # type: ignore[reportAttributeAccessIssue]
            return (self.translit == t_translit
                    or self.syriac == t_syriac)
        # else
        return value in {self.translit, self.syriac}


class Verse(Any):
    """Represents a verse in a biblical text.

    Attributes:
        book: Name of the biblical book the verse belongs to.
        reference: A verse reference, e.g. "Genesis Chapter 01 Verse 01"
        words: List of words represented as PeshittaWord objects

    .. caution::
        This class does not implement __hash__ method, which means that
        this class is unhashable and thus cannot be used as a key in ``dict``
        or as a member of ``set``. This is due to the way __eq__ comparator
        is implemented. Make sure to use one of the attributes as a key when
        using dict.

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

        Raises:
            ValueError: if the number of words in transliteration do not match
                with that of Syriac script or the number of word annotations.
        """
        # Initialise the class attrs
        self.book: str = book_name
        self.reference: str = verse_ref
        self.words: list[PeshittaWord] = []

        # Check for invalid arguments
        if (book_name is None or verse_ref is None or translit_words is None):
            msg = (f"book_name ({book_name}), verse_ref ({verse_ref}), and"
                    + f" translit_words ({translit_words}) given: "
                    + "they cannot be None!")
            raise ValueError(msg)

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
                        origin=origin,
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
                logger.warning("No origin provided to Verse.__init__")
                self.words.append(PeshittaWord(
                        translit_words[i],
                        ))

    def __eq__(self, value: object, /) -> bool:
        try:
            # reportAttributeAccessIssue can be ignored here since
            # AttributeError is explicitly handled.
            return (self.get_translit_words() == value.get_translit_words()  # type: ignore[reportAttributeAccessIssue]
                    or self.get_syriac_words() == value.get_syriac_words())  # type: ignore[reportAttributeAccessIssue]
        except AttributeError:
            return (value == self.get_translit_words()
                    or value == self.get_syriac_words())

    def __str__(self) -> str:
        """Return string representation of the verse.

        Returns:
            A string where transliterations words are joined with a space,
            creating a string containing the all words in the verse.
        """
        words = " ".join([w.translit for w in self.words])
        return self.reference + " | " + words

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
            :class:`src.classifier.result_utils.PeshittaWord`
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

        .. note::
            If the verse's syriac text is not registered upon initialisation,
            this function returns a list of empty strings with the length
            same as the number of words.

        .. seealso::
            :class:`src.classifier.result_utils.PeshittaWord`
                Refer to ``translit`` attribute there for details of
                Syriac script words.
        """
        return [w.syriac for w in self.words]

    def get_words_in_mode(self, mode: int = 1) -> list[str]:
        """Get the list of words in the script defined by ``mode``.

        Args:
            mode: defines the mode in which to parse the text.
                if ``1``, then use the transliteration.
                if ``2``, then use original Syriac script.

        Returns:
            list of words in the verse in Syriac script(s).


        .. note::
            If the verse's syriac text is not registered upon initialisation,
            this function returns a list of empty strings with
            the length same as the number of words when ``mode = syriac``.

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

        .. note::
            If annotations for words in the verse are not registered upon
            initialisation, this function returns a list of empty strings with
            the length same as the number of words.
        """
        return [w.annots for w in self.words]


class FileFormatterProto(Protocol):
    """A protocol that defines the interface of file formatter functions."""
    def __call__(self,
                samples: list[Verse] | NDArray[Verse],
                probas: list[list[float]] | NDArray[np.float64],
                correct_labels: list[int] | NDArray[np.int64] | None = None
                ) -> str:  # type: ignore[reportReturnType]
        """Defines the signature for a FileFormatterProto function.

        Args:
            samples: a list of samples (
                :class:`src.classifier.result_utils.Verse` instances) to save.
            probas: two-dimensional list of probabilities of each verse
                belonging to each of the classes.
            correct_labels: gold reference for the samples.

        Returns:
            string containing the prediction results in a certain file format.
        """


class Predictions(Any):
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
            samples: list[Verse] | NDArray[Verse],
            predictions: list[int] | NDArray[np.int64],
            correct_labels: list[int] | NDArray[np.int64] | None = None,
        ) -> None:
        self.samples = samples
        self.predictions = predictions
        self.correct_labels = correct_labels

        if len(self.samples) != len(self.predictions):
            msg = ("length of provided lists/arrays for samples and predictions"
            + " do not match.")
            raise ValueError(msg)
        if (self.correct_labels is not None
            and len(self.samples) != len(self.correct_labels)):
            msg = ("length of provided lists/arrays for samples and correct"
                + " labels do not match.")
            raise ValueError(msg)


class ProbaPredictions(Predictions):
    """A dataclass to hold prediction data with probabilities.

    Attributes:
        probas: probability of each sample belonging to each class, predicted
            by the classifier.

    .. seealso::
        See :class:`src.classifier.result_utils.Predictions` for
            other arguments.
    """
    def __init__(
            self,
            samples: list[Verse] | NDArray[Verse],
            predictions: list[int] | NDArray[np.int64],
            probas: list[list[float]] | NDArray[np.float64],
            correct_labels: list[int] | NDArray[np.int64] | None = None,
        ) -> None:
        """Initialises an instance.

        Raises:
            ValueError: if the lengths of provided arguments samples and probas
                do not match
        """
        if len(probas) != len(samples):
            msg = (f"lengths of probas {len(probas)} and samples {len(samples)}"
                    + " do not match!")
            raise ValueError(msg)
        super().__init__(samples, predictions,
                        correct_labels)
        self._probas = probas

    def get_probas(self) -> list[list[float]] | NDArray:
        """Get a list of predicted probabilities.

        Returns:
            list of probabilities, of size (number of samples,
                number of prediction classes)
        """
        return self._probas

    def get_total_probas(self) -> dict[str, list[float]]:
        """Get the total probabilities for the book being Christian or Jewish.

        Returns:
            a dict of book name as the key and lists as the associated value,
            where each sub-list contains [
            probability of the book being of Jewish authorship (label == 0),
            probability of the book being of Christian authorship (label == 1)
            ]
        """
        current_book = ""
        aj = 1.0  # total probability of the book's Jewish authorship
        ac = 1.0  # total probability of the book's Christian authorship
        smoothing_term = 1
        total_probas = {}
        for i in range(len(self.samples)):
            if current_book and self.samples[i].book != current_book:
                # if we finish walking through the verses from one book
                # calculate percentage of the "probabilities"
                pj = aj / (aj + ac)
                pc = ac / (aj + ac)
                append_to_dict(current_book, [pj, pc], total_probas)
            if self.samples[i].book != current_book:
                current_book = self.samples[i].book
                aj = 1.0
                ac = 1.0
            # perform smoothing to avoid 0.0 probas
            aj *= (self._probas[i][0] + smoothing_term)
            ac *= (self._probas[i][1] + smoothing_term)
        # calculate percentage of the "probabilities"
        pj = aj / (aj + ac)
        pc = ac / (aj + ac)
        append_to_dict(current_book, [pj, pc], total_probas)
        return total_probas

    def save_to_file(self,
                     formatter: FileFormatterProto,
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
        if self.correct_labels is not None:
            data_str = formatter(samples=self.samples,
                                        probas=self.get_probas(),
                                        correct_labels=self.correct_labels
                                 )
        else:
            data_str = formatter(samples=self.samples,
                                        probas=self.get_probas())

        # write formatted texts to files
        Path(save_file).write_text(data_str, encoding="utf-8")


class Mislabels(Any):
    """Wrapper of mislabelled results summary to facilitate human inspection.

    An instance of this class represents a group of mislabelled samples
    from a particular attribution class, or in this case,
    one of the two testaments.

    Attributes:
        incorrect_labels: incorrect labels the classifier assigned
        correct_labels: gold references for the verses
        mislabelled_verses: mislabelled verses corresponding
        probas: probabilities of the mislabelled verses belonging to each class,
            predicted by the src.classifier. Defaults to None if not provided
            upon initialisation.
    """
    def __init__(
            self,
            incorrect_labels: list[int] | NDArray[np.int64],
            correct_labels: list[int] | NDArray[np.int64],
            mislabelled_verses: list[Verse],
            probas: list[list[float]] | NDArray[np.float64]
        ) -> None:

        if (len(mislabelled_verses) < 1
            or len(incorrect_labels) < 1
            or len(correct_labels) < 1):
            msg = ("indices of mislabelled samples were empty."
                   + "Make sure to instanciate this only when "
                   + "there are one or more mislabelled samples.")
            raise ValueError(msg)

        self.mislabels = incorrect_labels
        self.correct_labels = correct_labels
        self.verses: list[Verse] = mislabelled_verses
        self.probas = probas

    def __len__(self) -> int:
        """An under-the-hood method defining the result of :func:`len`.

        Returns:
            An integer indicating number of mislabelled verses.
        """
        return len(self.mislabels)

    def save_to_file(
            self,
            formatter: FileFormatterProto,
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
        assert (len(self.verses) == len(self.probas))
        save_data = formatter(
            samples=self.verses,
            probas=self.probas,
            correct_labels=self.correct_labels
        )

        # write formatted texts to files
        if isinstance(save_file, str):
            save_file_p = Path(clean_path_str(save_file))
        else:
            save_file_p = save_file
        save_file_p.write_text(save_data, encoding="utf-8")
