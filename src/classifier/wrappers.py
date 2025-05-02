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

"""Classes for general pipeline representation."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Self

from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.classifier.fitting_utils import make_feature, make_vocab, vectorise
from src.classifier.result_utils import Verse


class Classifier(ClassifierMixin, BaseEstimator):
    """General representation of a classifier interface.

    Attributes:
        algo: Learning algorithm to use. This class expects a model for
            binary classification.
        train_vector: the vectorised representation of the training data.
        train_y: the gold reference labels for the training data.
    """

    def __init__(self, clf: BaseEstimator) -> None:
        """Initialise a BoWEstimator class instance.

        Args:
            clf: this can be any :class:`BaseEstimator` object as long as
            they have ``.fit`` and ``.predict`` methods.

        Raises:
            ValueError: if ``clf`` does not have methods named ``.fit``
                and ``.predict``.
        """
        # validate that the clf argument has .fit and .predict methods
        if not (hasattr(clf, "fit") and callable(clf.fit)):  # type: ignore[reportArgumentType]
            msg = f"clf {clf} does not have a method named `.fit`"
            raise ValueError(msg)
        if not (hasattr(clf, "predict") and callable(clf.predict)):  # type: ignore[reportArgumentType]
            msg: str = f"clf {clf} does not have a method named `.predict`"
            raise ValueError(msg)

        # Initialise instance variables
        # AttributeAccessIssue can be ignored here since we explicitly check
        # for necessary classes above (so this can be duck typed)
        self.algo: Callable = clf  # type: ignore[reportAttributeAccessIssue]
        self.train_vector: Sequence[Sequence[int | float]] = []
        self.train_y: list[int] = []

    def fit(
        self,
        X: list | np.ndarray,
        y: list[int] | None = None,
    ) -> Self:
        """Fit a model on the provided data.

        Args:
            X: training samples to fit the model. This naming contradicts
            the PEP, but it's defined so to comply with the scikit-learn's
            conventions.
            y: correct labels (gold reference) of the training samples.

        Returns:
            A Classifier class instance after fitting. This behaviour mimics
            scikit-learn's classifiers.
        """
        if y is None:
            return self.algo.fit(X)  # type: ignore[reportArgumentType]
        return self.algo.fit(X, y)  # type: ignore[reportArgumentType]

    def predict(self, X: list | np.ndarray) -> list | np.ndarray:
        """Predict on the provided data with the model.

        Args:
            X: some samples to predict using the model. This naming contradicts
            the PEP, but it's defined so to comply with the scikit-learn's
            conventions.

        Returns:
            List of prediction result of each sample.
        """
        return self.algo.predict(X)  # type: ignore[reportArgumentType]


class ProbaClassifier(Classifier):
    """General representation of a classifier with "predict_proba" method.

    Attributes:
        algo: Learning algorithm to use. This class expects a model for
            binary classification.
        train_vector: the vectorised representation of the training data.
        train_y: the gold reference labels for the training data.
    """

    def __init__(self, clf: BaseEstimator) -> None:
        # validate that clf has ``.predict_proba`` method
        if not (
            hasattr(clf, "predict_proba") and callable(clf.predict_proba)  # type: ignore[reportArgumentType]
        ):
            msg = f"clf {clf} does not have a method named `predict_proba`."
            raise ValueError(msg)
        super().__init__(clf)

    def predict_proba(
        self, X: list | NDArray
    ) -> list[list[float]] | NDArray:
        """Predict probabilities on the provided samples.

        Args:
            X: The sample to predict probabilities on.

        Returns:
            A two-dimensional list of flaots, each subarray representing one
            sample in ``X`` and each coordinate ``X[i]`` holds a probability of
            the sample belonging to class ``i``.
        """
        return self.algo.predict_proba(X)  # type: ignore[reportArgumentType]


class BoWEstimator(ProbaClassifier):
    """Wrapper around the BoW vectorisation to simplify training and testing.

    Attributes:
        algo: Learning algorithm to use. This class expects a model for
            binary classification.
        n: the size of window for n-gram extraction, i.e. *N* of n-grams.
        n_gram_formatter: string preprocessing function before extracting
            n-grams. Decides if the model uses word n-grams or character
            n-grams.
        vocabs: model vocabulary and verse-level n-gram counts constructed
            with the :func:`src.classifier.fitting_utils.make_vocab`
        train_vector: the vectorised representation of the training data.
        train_y: the gold reference labels for the training data.
        pred_vector: the vectorised representation of the data to predict.

    .. seealso::
        :func:`src.classifier.fitting_utils.count_n_grams`
            for attrs: ``n``, ``n_gram_formatter``.
    """

    def __init__(
        self,
        clf: BaseEstimator,
        formatter: Callable,
        n: int = 3,
    ) -> None:
        """Initialise a BoWEstimator class instance.

        Args:
            clf: this can be any :class:`BaseEstimator` object as long as they
            have ``.fit`` and ``.predict`` methods.
            formatter: a :class:`Callable` object to format the verse string
                before applying :func:`nltk.util.ngrams` function.
            n: ``n`` of n-grams.

        .. seealso::
            :func:`src.classifier.fitting_utils.make_vocab`
        """
        super().__init__(clf)
        self.n: int = n
        self.n_gram_formatter: Callable = formatter
        self.vocabs: tuple[Counter, list[Counter]] | None = None
        self.train_vector: Sequence[Sequence[int | float]] = []
        self.train_y: list[int] = []
        self.pred_vector: np.ndarray | list[list[int]] | None = None
        self.preprocessor: Callable[[list[str]], list[str]] | None = None

    def set_preprocessor(
        self, preprocessor: Callable[[list[str]], list[str]]
    ) -> None:
        """Register a global preprocessor.

        Args:
            preprocessor: a callable object to preprocess all string before
                training and testing (prediction).

        .. note:
            Since a preprocessor function takes
            a list of plain str objects, it should only deal with sub-word-level
            features (i.e. specific characters). Higher-level text manipulation,
            such as removing words of a particular POS, must be performed before
            the samples are passed to ``.fit``, ``.predict``, or
            ``.predict_proba`` methods.
        """
        self.preprocessor = preprocessor

    def _process_verses(
        self, X: list[Verse] | list[list[str]]
    ) -> list[list[str]]:
        verses = []
        for vrs in X:
            current = ""
            if isinstance(vrs, Verse):
                current = vrs.get_translit_words()
            else:
                current = vrs
            # Apply preprocessing function if registered
            if self.preprocessor is not None:
                verses.append(self.preprocessor(current))
            else:
                verses.append(current)
        return verses

    def fit(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        X: list[Verse] | list[list[str]],
        y: list[int],
    ) -> None:
        """Train the algorithm on train_x to get a classifier.

        .. note:: This method is in strict terms incompatible with the super
            classes' ``.fit`` method because ``list[Verse]`` is incompatible
            with ``np.typing.NDArray`` in some cases.

        Args:
            X: the samples to train the model.
            y: a one-dimensional list of labels for each sample.

        Raises:
            ValueError: if the length of model training input train_x and
            reference labels y do not match.
            RuntimeError: if the return value of make_vocab is None.
        """
        if len(X) != len(y):
            msg = f"Length of X ({len(X)}) and y ({len(y)}) do not match."
            raise ValueError(msg)

        self.vocabs = make_vocab(
            self._process_verses(X), self.n_gram_formatter, span=self.n
        )
        self.train_y = y

        if self.vocabs is None:
            msg = (
                "The result of `make_vocab` was None. Failed to construct"
                + " vocabulary to fit the model. Aborting."
            )
            raise RuntimeError(msg)

        print(
            f"Found {len(self.vocabs[0])} independent n-grams "
            + f"from {len(self.vocabs[1])} verses!"
        )
        self.train_vector = make_feature(self.vocabs)
        self.algo.fit(self.train_vector, y)  # type: ignore[reportArgumentType]

    def predict(self, X: list[Verse] | list[list[str]]) -> NDArray:  # type: ignore[reportIncompatibleMethodOverride]
        """Predict on target_x with the pretrained classifier.

        .. note:: This method is in strict terms incompatible with the super
            classes' ``.fit`` method because ``list[Verse]`` is incompatible
            with ``np.typing.NDArray`` in some cases.

        Args:
            X: list of inputs to the model

        Returns:
            list of predicted class labels.

        Raises:
            ValueError: if ``self.vocabs`` is still empty. This is likely
                because one forgot to run `.fit` method before running
                `.predict`.
        """
        if self.vocabs is None:
            msg = (
                "Cannot fetch the vocabulary of the model. "
                + "You must run `.fit` method before making predictions."
            )
            raise ValueError(msg)
        if len(X) == 0:
            msg = "X is empty. You must provide samples to predict."
            raise ValueError(msg)

        self.pred_vector = np.array(
            vectorise(
                self._process_verses(X),
                self.vocabs[0],
                self.n_gram_formatter,
                span=self.n,
            )
        )
        return self.algo.predict(self.pred_vector)  # type: ignore[reportAttributeAccessIssue]

    def predict_proba(  # type: ignore[reportIncompatibleMethodOverride]
        self, X: list[Verse] | list[list[str]]
    ) -> np.ndarray:  # type: ignore[reportIncompatibleMethodOverride]
        """Predict probability on target_x with the pretrained classifier.

        .. note:: This method is in strict terms incompatible with the super
            classes' ``.fit`` method because ``list[Verse]`` is incompatible
            with ``np.typing.NDArray`` in some cases.

        Args:
            X: list of input samples to the model

        Returns:
            A two-dimensional list of flaots, each subarray representing one
            sample in ``X`` and each coordinate ``X[i]`` holds a probability of
            the sample belonging to class ``i``.

        Raises:
            ValueError: if X is empty. also raised when ``self.vocabs`` is
                not set. This is likely because one forgot to run `.fit` method
                before running `.predict`.
        """
        if self.vocabs is None:
            msg = (
                "Cannot fetch the vocabulary of the model. "
                + "You must run `.fit` method before making predictions."
            )
            raise ValueError(msg)
        if len(X) == 0:
            msg = "X is empty. You must provide samples to predict_proba."
            raise ValueError(msg)

        targets = np.array(
            vectorise(
                self._process_verses(X),
                self.vocabs[0],
                self.n_gram_formatter,
                span=self.n,
            )
        )

        return self.algo.predict_proba(targets)  # type: ignore[reportFunctionMemberAccess]
