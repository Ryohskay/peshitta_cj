"""Definition of skeleton dictionary for datasets.

.. attention:: When using this skeleton, always ``.copy()`` them so you don't
    accidentally mess with some other datasets loaded using this format.
"""


from classifier.result_utils import Verse


class DataSplit:
    """An abstract representation of a group of data within one dataset.

    This helps splitting the dataset into the train, test, production sets.

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


class LoadedDataset:
    """A wrapper around datasets loaded from files."""
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
