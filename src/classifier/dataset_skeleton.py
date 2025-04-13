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

"""Definition of skeleton dictionary for datasets.

.. attention:: When using this skeleton, always ``.copy()`` them so you don't
    accidentally mess with some other datasets loaded using this format.
"""

import json
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Literal, TypedDict

from src.classifier.result_utils import Verse


class DatasetDict(TypedDict):
    """An entry in a dataset format for Huggingface."""
    label: int | str
    text: str


class DataSplit:
    """An abstract representation of subset of data within one dataset.

    Attributes:
        num_classes: the number of target categories you want this system to
            classify.
        verses: a list containing ``num_classes`` lists of verses for each
            class. In this project, ``num_classes = 2``, so there are 2 lists.
            The list at the index 0 represents the verses from OT, and the one
            at the index 1 represents verses from NT. These indices must match
            the labels assigned for each of the classes.

    .. note::
        internally, this class stores data in a list of lists, where each
        sub-list at index ``i`` contains the instances of class ``i``.
    """
    def __init__(self,
                 ot_verses: list[Verse],
                 nt_verses: list[Verse]
             ) -> None:
        self.num_classes: int = 2  # this is a binary classification problem
        self.verses: list[list[Verse]] = []
        self.verses.append(ot_verses)
        self.verses.append(nt_verses)

    def _validate_target(self, t: int | None) -> int | None:
        """Validate if provided integer for label is in the dataset.

        Args:
            t: None or an integer value, the label specifying a class
                in the dataset.

        Returns:
            the input value *as is*.

        Raises:
            ValueError: if ``t`` is not None but out of the range of
                expected integer labels.
        """
        if t is not None and t not in range(self.num_classes):
            msg = (f"Integer label {t} cannot be generated. "
                   + "They are not in the expected labels for this system."
                   + " Please refer to the documentation for: "
                   + "datasets.DataSplit")
            raise ValueError(msg)
        return t

    def get_samples(self,
                    target: int | None = None
                ) -> list[Verse]:
        """Get samples or instances of each class.

        Args:
            target: By default, returns a list all instances of all classes
                in this split as a one-dimensional list as
                ``self.verses[0][0], ... self.verses[i][j]`` where ``i``
                is the (number of classes - 1) and [j] is (the number of samples
                for the class ``i`` - 1).
                If target is a valid integer, returns a list from
                ``self.verses[target]``.

        Returns:
            a list of the samples from the class ``target``.
        """
        target = self._validate_target(target)
        # default behaviour
        if target is None:
            all_verses = self.verses[0].copy()
            for i in range(1, self.num_classes):
                all_verses.extend(self.verses[i])
            return all_verses
        # else
        return self.verses[target]

    def get_labels(self,
                   target: int | None = None
                ) -> list[int]:
        """Generate labels for the specified targets.

        Args:
            target: By default, returns a list of labels for all instances
                in this split as a one-dimensional list as
                ``self.verses[0][0], ... self.verses[i][j]`` where ``i``
                is the (number of classes - 1) and [j] is (the number of samples
                for the class ``i`` - 1).
                If target is a valid integer, returns a list of ``target``
                integer repeated for the length of ``self.verses[target]``.

        Returns:
            a list of integers labelling the instances of the class ``target``.
        """
        target = self._validate_target(target)
        # default behaviour
        if target is None:
            # generate a list of labels for all instances in this split
            # by concatenating the labels for each class
            labels = [0 for v in self.verses[0]]
            for i in range(1, self.num_classes):
                labels.extend([1 for v in self.verses[i]])
            return labels
        # else
        return [target for v in self.verses[target]]

    def generate_syriac_hf(self, target: int | None = None) -> Generator[DatasetDict]:
        """Generate the syriac text of verses in Huggingface datasets format.

        Each yielded item is a JSONL (JSON Lines) line.

        Args:
            target: integer designating the label for the target class to
                extract.

        .. seealso::
            :meth:`src.classifier.dataset_skeleton.DataSplit.get_labels` for a
            more thorough description of the argument `target`.

        Yields:
            A :class:`src.classifier.dataset_skeleton.DatasetDict` instance,
            representing one entry in a dataset. For Huggingface datasets
            library, write this to a file as a line and give it a ``.json``
            file extension.
        """
        if target is None:
            for i in range(self.num_classes):
                verses = self.get_samples(i)
                for v in verses:
                    yield {"label": i,
                            "text": " ".join(v.get_syriac_words())}
        else:
            label: int = self._validate_target(target)  # type: ignore[reportAssignmentType]
            verses = self.get_samples(label)
            for v in verses:
                yield {"label": label,
                        "text": " ".join(v.get_syriac_words())}

    def generate_translit_hf(self, target: int | None = None
                             ) -> Generator[DatasetDict]:
        """Generate the transliterated verses in Huggingface datasets format.

        Each yielded item is a JSONL (JSON Lines) line.

        Args:
            target: integer designating the label for the target class to
                extract.

        .. seealso::
            :meth:`src.classifier.dataset_skeleton.DataSplit.get_labels` for a
            more thorough description of the argument `target`.

        Yields:
            A :class:`src.classifier.dataset_skeleton.DatasetDict` instance,
            representing one entry in a dataset. For Huggingface datasets
            library, write this to a file as a line and give it a ``.json``
            file extension.
        """
        stop = self.num_classes
        if target is not None:
            stop: int = self._validate_target(target) + 1  # type: ignore[reportAssignmentType]
        for i in range(stop):
            verses = self.get_samples(target)
            for v in verses:
                yield {"label": i,
                        "text": " ".join(v.get_syriac_words())}


class LoadedDataset:
    """A wrapper around datasets loaded from files.

    Attributes:
        train: a :class:`src.classifier.dataset_skeleton.DataSplit` instance
            containing the train split (training data).
        test: a :class:`src.classifier.dataset_skeleton.DataSplit` instance
            containing the test split (test data).

        .. note::
            Use the cross-validation technique or create a child class
            inheriting this ``LoadedDataset`` class if you want a separate
            evaluation data split.
    """
    def __init__(
            self,
            ot_train_verses: list[Verse],
            nt_train_verses: list[Verse],
            ot_test_verses: list[Verse],
            nt_test_verses: list[Verse],
            production_verses: list[Verse] | None
        ) -> None:
        self.train = DataSplit(ot_train_verses, nt_train_verses)
        self.test = DataSplit(ot_test_verses, nt_test_verses)
        if production_verses is not None:
            self.production = production_verses

    def save_as_json(
            self,
            save_dir: str | Path,
            mode: Literal["syriac", "translit"] = "syriac",
        ) -> None:
        """Save the given data in json compatible with Huggingface datasets.

        Args:
            save_dir: a string or :class:`python:pathlib.Path` object for
                the path to save the dataset files.
            mode: integer indicating the label for the class. 

        Raises:
            FileNotFoundError: when the directory for saving the dataset files
                could not be found.
        """
        save_dir_p = Path(save_dir)
        if not save_dir_p.exists():
            raise FileNotFoundError

        # save train dataset
        for i in range(self.train.num_classes):
            verse_data = list(self.train.generate_hf(i))

            save_file = save_dir_p / f"etcbc_train_data_{i}.json"
            with save_file.open("w", encoding="utf-8") as fp:
                json.dump(verse_data, fp)

            print(f"Saved: {save_file.resolve()}")

        # save test dataset
        for i in range(self.test.num_classes):
            verse_data = list(self.test.generate__hf(i))

            save_file = save_dir_p / f"etcbc_test_data_{i}.json"
            with save_file.open("w", encoding="utf-8") as fp:
                json.dump(verse_data, fp)

            print(f"Saved: {save_file.resolve()}")

        # save production dataset
        prod_verses = []
        for verse in self.production:
            prod_verses.append(
                    {"text": f"{verse.get_syriac_words()}"}
                    )

        save_file = save_dir_p / "etcbc_production_data.json"
        with save_file.open("w", encoding="utf-8") as fp:
            json.dump(prod_verses, fp)

        print(f"Saved: {save_file.resolve()}")
