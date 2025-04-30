"""Utilities to construct file names."""

import re
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Iterable, Literal, Self

from src.shared import label_data


class FnameExtraOpts(Enum):
    """An Enum of possible extra file name options."""

    REMOVE_DIACRITICS = "_no_diacritics"
    REMOVE_PROPN = "_no_propn"
    REMOVE_UNDERSCORES = "_no_uscore"
    REMOVE_FROM_BOTH = "_both_removed"  # removed from both train and test sets
    # erroneous data which should not be taken seriously
    IS_ERRONEOUS = "_ERRONEOUS"


def _sanitise(s: str) -> str:
    """Return a str where special path characters are removed from s."""
    # clean the string and remove dots to avoid confusing file names
    s = s.strip().lower().replace(r".", "")
    # replace every non-alnum char except underscore with an underscore
    return re.sub(r"\W", "_", s)


class SavefileName():
    """A class to handle and construct a save file's name."""

    def __init__(
        self,
        origin: Literal["CAL", "ETCBC"],
        classifier_alias: str,
        file_ext: str = "csv",
    ) -> None:
        """Initialise the skeleton for the savefile name.

        Args:
            origin: the origin of the samples.
            classifier_alias: an alias to the classifier. E.g. 'mnb' for
                ``MultinomialNB``.
            extra_suffix: any suffix to add at the end of the file name,
                before the file extension.
            file_ext: file extension to append at the end of the file name.

        Raises:
            ValueError: if any of data origin, file extension, or classifier
                alias is given but empty or invalid.
        """
        # check for erroneous options or such combinations
        if not origin:
            msg = "Data origin is empty!"
            raise ValueError(msg)
        if origin not in {"ETCBC", "CAL"}:
            msg = f"Unknown data origin: {origin}"
            raise ValueError(msg)

        self.origin: Literal["CAL", "ETCBC"] = origin

        if not classifier_alias:
            msg = "Classifier alias is empty!"
            raise ValueError(msg)
        if not file_ext:
            msg = "File extension is empty!"
            raise ValueError(msg)

        self.classifier = _sanitise(classifier_alias)
        self.ext = _sanitise(file_ext)

        # initialise empty attributes
        self.n = 0
        self.is_char_level = False
        self.is_n_gram = False
        self.is_bow = False
        self.scope = ""
        # other per-algorithm features
        # k-nearest neighbours
        self.knn_k: int = 0
        self.knn_weights: Literal["uniform", "distance", ""] = ""
        self.knn_minkowski_p: int = 0
        # random forest
        self.n_estimators: int = 0
        # multi-layer perceptron
        self.hidden_layer_sizes: list[int] = []
        self.activation: str = ""  # relu, tanh, logistic, identity
        # Special file flags
        self.is_prod = False
        self.is_mislabel = False
        self.is_total_proba = False
        self.is_clf_summary = False
        # initialise extra options
        self.extra_opts: dict[FnameExtraOpts, bool] = {}
        for opt in FnameExtraOpts:
            self.extra_opts[opt] = False

    def is_same_classifier(self, obj: Self) -> bool:
        """Check if results in two save files are from the same classifier.

        This function compares the classifier configurations represented in two
        save files' names, regardless of the file contents.
        E.g. a total probability file and a mislabel file are evaluated to
        ``True`` if the results come from the same classifier even if their
        contents and/or formats differ.

        Args:
            obj: another :class:`SavefileName` instance.

        Returns:
            a boolean indicating if the classifiers are different.
        """
        if not isinstance(obj, SavefileName):
            return False
        return (
            self.origin == obj.origin
            and self.classifier == obj.classifier
            and self.is_n_gram == obj.is_n_gram
            and self.is_char_level == obj.is_char_level
            and self.is_bow == obj.is_bow
            and self.n == obj.n
            and self.knn_k == obj.knn_k
            and self.knn_weights == obj.knn_weights
            and self.knn_minkowski_p == obj.knn_minkowski_p
            and self.n_estimators == obj.n_estimators
            and self.hidden_layer_sizes == obj.hidden_layer_sizes
            and self.activation == obj.activation
            and self.extra_opts == obj.extra_opts
        )

    def as_path(self) -> Path:
        """Return the file name as a :class:`python:pathlib.Path` instance."""
        return Path(self.get_fname())

    def set_ngram_opts(
        self,
        n: int = 3,
        *,
        is_char_level: bool = True,
        is_n_gram: bool = True,
        is_bow: bool = True,
    ) -> None:
        """Set the options related to classifiers based on n-gram models.

        Args:
            n: the ``n`` of n-grams. this value is only used if the arg
                ``is_n_gram`` is set to ``True``.
            is_char_level: a boolean indicating if the classifier is a character
                level model. ``False`` indicates a word-level model. this value
                is only used if the arg ``is_n_gram`` is set to ``True``.
            is_n_gram: a boolean indicating if the classifier uses an n-gram
                in tokenisation.
            is_bow: a boolean indicating if the classifier uses a Bag-of-Words
                approach. this value is only used if the arg ``is_n_gram`` is
                set to ``True``.
        """
        self.is_n_gram = is_n_gram
        if self.is_n_gram:
            self.n = int(n)
            self.is_bow = is_bow
            self.is_char_level = is_char_level

    def set_scope(self, scope: str) -> None:
        """Set the scope of the results stored in the file.

        Args:
            scope: a str designating the scope. This must be a valid name
                associated with a label in :mod:`src.shared.label_data`.
        """
        if scope not in label_data.LabelToVal:
            msg = f"Invalid scope or label name: {scope}"
            raise ValueError(msg)
        self.scope = _sanitise(scope)

    def add_extra_opts(
        self, extra_opts: list[FnameExtraOpts] | None = None
    ) -> None:
        """Append extra optional element(s) to the file name.

        Args:
            extra_opts: any enum object from the ``FnameExtraOpts``
                or its value.

        Raises:
            ValueError: if no option is given, or if any of the given options
                is invalid.
        """
        if extra_opts is None or len(extra_opts) == 0:
            msg = "No option provided!"
            raise ValueError(msg)

        for opt in extra_opts:
            # check if all provided options are valid
            if opt not in FnameExtraOpts:
                # NB: simply evaluating ``in`` upon the enum class
                # does not suffice as a value in FnameExtraOpts will
                # also be evaluated to ``True``.
                msg = f"Invalid option: {opt}"
                raise ValueError(msg)
            # register the extra option
            self.extra_opts[opt] = True


    def set_knn_opts(
        self,
        k: int = 5,
        *,
        weights: Literal["uniform", "distance"] = "uniform",
        p: int = 2,
    ) -> None:
        """Set the options related to classifiers based on k-nearest neighbours.

        Args:
            k: the number of neighbours to consider.
            weights: the weight function used in prediction. Possible values are
                ``uniform`` and ``distance``. ``uniform`` means all points
                are weighted equally, while ``distance`` means closer points
                have more influence on the prediction.
            p: the power parameter for the Minkowski distance metric.
        """
        self.knn_k = int(k)
        self.knn_weights = weights.lower()  # type: ignore[reportArgumentType]
        self.knn_minkowski_p = int(p)

    def set_rf_opts(self, n_estimators: int = 100) -> None:
        """Set the options related to classifiers based on random forests.

        Args:
            n_estimators: the number of trees in the forest.
        """
        self.n_estimators = int(n_estimators)

    def set_mlp_opts(
        self,
        hidden_layer_sizes: Iterable[int] = (100,),
        *,
        activation: Literal[
            "identity",
            "logistic",
            "tanh",
            "relu",
        ] = "relu",
    ) -> None:
        """Set the options related to classifiers based on multi-layer perceptron.

        Args:
            hidden_layer_sizes: the number of neurons in each hidden layer.
            activation: the activation function for the hidden layer.
                Possible values are ``identity``, ``logistic``, ``tanh``, and
                ``relu``.

        Raises:
            ValueError: if the name of the activation function is invalid.
        """
        activation_f = activation.lower()
        if activation_f not in {"relu", "tanh", "logistic", "identity"}:
            msg = f"The name of activation function {activation_f} is invalid."
            raise ValueError(msg)
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.activation = activation_f

    def copy(self, memo: dict | None = None) -> Self:
        """Returns a deep copy of self."""
        return deepcopy(self, memo)

    def mark_special_file(
        self,
        *,
        is_mislabel: bool = False,
        is_prod: bool = False,
        is_total_proba: bool = False,
        is_clf_summary: bool = False,
    ) -> None:
        """Mark the file as a special type of data save file.

        Args:
            is_mislabel: a boolean indicating if the file stores the mislabelled
                verses, written by the method
                :meth:`src.classifier.result_utils.Mislabels.save_to_file`.
            is_prod: a boolean indicating if the file is for the predictions
                on the production dataset.
            is_total_proba: a boolean indicating if the file is for the
                total probability of classification per book.
            is_clf_summary: a boolean indicating if the file is for the summary
                of a particular classifier.
        """
        if is_mislabel and is_prod:
            msg = (
                "Mislabelled verse is undefined for production data"
                + " because you don't know the correct labels!"
            )
            raise ValueError(msg)
        if is_mislabel and is_total_proba:
            msg = (
                "The file cannot be a mislabelling inspection data AND per-book"
                + " total probability data at the same time!"
            )
            raise ValueError(msg)
        if is_clf_summary and (is_mislabel or is_total_proba):
            msg = (
                "The classifier summary file cannot be mislabel file nor "
                + "per-book total probability data file!"
            )
            raise ValueError(msg)
        self.is_prod = is_prod
        self.is_mislabel = is_mislabel
        self.is_total_proba = is_total_proba
        self.is_clf_summary = is_clf_summary

    def get_fname(  # noqa: C901, PLR0912
        self,
    ) -> str:
        """Get a file name to save the classifier prediction results.

        Returns:
            a str of the constructed file name.

        Raises:
            ValueError: if the combination of options are impossible
                by definition.
        """
        # construct the file name
        fname = _sanitise(self.origin)
        fname += "_" + self.classifier

        # append options related to n-gram models
        if self.is_n_gram:
            if self.is_char_level:
                fname += f"_char_{self.n}gram"
            else:
                fname += f"_word_{self.n}gram"

        if self.is_bow:
            fname += "_bow"

        # append options related to k-nearest neighbours
        if self.knn_k != 0:
            fname += f"_{self.knn_k}knn"
            if self.knn_weights:
                fname += "_" + self.knn_weights
            if self.knn_minkowski_p:
                fname += f"_p{self.knn_minkowski_p}"

        # append options related to random forests
        if self.n_estimators != 0:
            fname += f"_{self.n_estimators}rf"
        # append options related to multi-layer perceptron
        if len(self.hidden_layer_sizes) > 0:
            fname += f"_{len(list(self.hidden_layer_sizes))}layers"
            for layer in self.hidden_layer_sizes:
                fname += f"_{layer}"
            fname += "perceps"
            if self.activation:
                fname += "_activate_" + self.activation

        if self.scope:
            fname += "_" + self.scope

        # append extra options if defined
        # since self.extra_opts is a dict, it is guaranteed that the file name
        # options are always listed in the same order
        for opt in self.extra_opts:
            if self.extra_opts[opt]:
                fname += opt.value

        # add more suffixes based if the file contains a special kind of data
        if self.is_prod:
            fname = "PRODUCTION_" + fname

        if self.is_mislabel:
            fname += "_mislabels"  # contains only the mislabelled ones
        elif self.is_total_proba:
            # special file for the per-book total probas
            fname += "_total_proba"
        elif self.is_clf_summary:
            # special file for the classifier statistics summary
            fname += "_classifier_stats"
        else:
            fname += "_prediction_all"  # contains all predictions

        # add file extension
        fname += "." + self.ext
        return fname
