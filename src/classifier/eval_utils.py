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

"""Tools and functions to evaluate classifiers."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.classifier.dataset_skeleton import LoadedDataset
from src.classifier.prediction_utils import predict_proba
from src.classifier.result_utils import (
    Mislabels,
    Predictions,
    ProbaPredictions,
    Verse,
)
from src.classifier.wrappers import BoWEstimator


def mislabel_stats(
        inputs: np.ndarray | list[Any],
        y_correct: np.ndarray,
        preds: ProbaPredictions,
    ) -> Mislabels:
    """Calculate and print some basic statistics on model inputs & outputs.

    Args:
        inputs: Inputs passed when the classifier
            produced the predictions.
        y_correct: Correct (gold-standard) labels
            for the input verses.
        preds: a :class:`src.classifier.result_utils.Predictions` instance for
            the results of predictions on the input. When a
            :class:`src.classifier.result_utils.ProbaPredictions` instance is
            passed, the returned ``Mislabels`` class instance will have
            the ``probas`` attribute set.

    Returns:
        A :class:`src.classifier.result_utils.Mislabels` instance.

    Raises:
        ValueError: If the provided list of inputs to test is either zero
            length or does not have a measurable length.
    """
    sample_size = 0
    if hasattr(inputs, "shape"):
        # AttributeAccessIssue can be ignored since AttributeError is
        # implicitly handled by hasattr
        sample_size = int(inputs.shape[0])  # type: ignore[reportAttributeAccessIssue]
    else:
        # If the argument ``inputs`` is not a np.ndarray,
        # use the standard len() function instead.
        sample_size = len(inputs)

    if sample_size == 0:
        msg = ("Cannot measure the size of `inputs`!"
               + " It seems like the argument `inputs` is empty.")
        raise ValueError(msg)

    mislabel_books = {}
    current_book = ""
    num_book_verses = 0  # number of verses in the current book
    incorrect_labels = []
    correct_labels = []
    num_book_unk = 0  # number of verses labelled as unknown (-1)
    pred_probas = preds.get_probas()
    mislab_probas = []

    # count the number of mislabels per verse while counting the number of
    # verses in each book
    for i in range(len(preds.predictions)):
        num_book_verses += 1
        if (current_book != preds.samples[i].book):
            # if current_book is not empty,
            # calculate some statistics and print
            if current_book:
                num_mislabels = len(mislabel_books[current_book])
                print(f">> {current_book}: {num_mislabels} "
                      + "mislabelled verses, accounting for "
                      + f"{num_mislabels / num_book_verses:.02f}% out of "
                      + f"{num_book_verses} verses ({num_book_unk} verses "
                      + "labelled unknown)")
            # update the book-level variables
            current_book = preds.samples[i].book
            num_book_verses = 0
            num_book_unk = 0

        # the following section is triggered both when the above `if` section
        # is triggered and when it is not triggered but the verse's mislabelled
        if y_correct[i] != preds.predictions[i]:
            # print(preds.samples[i].book)
            # update variables other than book-level ones
            incorrect_labels.append(preds.predictions[i])
            correct_labels.append(y_correct[i])
            mislab_probas.append(pred_probas[i])

            # after updating the variables,
            # add the mislabelled verse to the mislabel_books dict
            if preds.samples[i].book not in mislabel_books:
                mislabel_books.update({
                        current_book: [preds.samples[i]]
                   })
            else:
                mislabel_books[current_book].append(preds.samples[i])
            if preds.predictions[i] == -1:
                num_book_unk += 1

    if current_book:
        num_mislabels = len(mislabel_books[current_book])
        print(f">> {current_book}: {num_mislabels} "
              + "mislabelled verses, accounting for "
              + f"{(num_mislabels / num_book_verses) * 100:.02f}% out of "
              + f"{num_book_verses} verses ({num_book_unk} verses "
              + "labelled unknown)")

    # Collate the collected mislabelling results into a Mislabels class instance
    mislabs = Mislabels(incorrect_labels, correct_labels,
    [mislab for book in mislabel_books
                                for mislab in mislabel_books[book]],
                        mislab_probas
              )

    # Print some stats
    print(
        f"Number of mislabelled points out of the total {sample_size} "
                + f"verses: {len(mislabs)} "
                + f"({incorrect_labels.count(-1)} labelled as unknown)"
    )
    print("Local accuracy: "
          + f"{accuracy_score(y_correct, preds.predictions):.02f}")

    return mislabs


def metricise(
        y_true: list[int] | NDArray,
        y_all: list[int] | NDArray,
        y_probas: list[list[float]] | NDArray | None = None
    ) -> tuple[float, NDArray, NDArray, NDArray]:
    """Calculate performance metrics of a classifier using the outputs.

    Args:
        y_true: Correct (gold reference) labels
        y_all: all predictions in a one-dimensional list.
        y_probas: probabilities predicted for each sample.

    Returns:
        accuracy (value), precision (1D array), recall (1D array),
        and f1 scores (1D array) as floats.

    Raises:
        ValueError: if the length of ``y_all`` after :func:`numpy.array` and
            the length of ``y_true`` do not match
    """
    y_pred = np.array(y_all)

    if len(y_pred) != len(y_true):
        msg = f"y_pred has length of {len(y_pred)}, but y_true {len(y_true)}"
        raise ValueError(msg)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, fbeta, support = precision_recall_fscore_support(
                    y_true, y_pred)

    print(f"Overall Accuracy: {accuracy:.02f}")
    print(f"Precisions: {precision}")
    print(f"Recalls: {recall}")
    print(f"F1 Score: {fbeta}")
    print(f"Supports: {support}")

    if y_probas is not None and len(y_probas) != len(y_true):
        msg = (f"length of y_probas {len(y_probas)} is not equal to "
               + f"length of y_true {len(y_true)}")
        raise ValueError(msg)

    if y_probas is not None:
        # calculate stats that use probabilities
        y_probas_pos = [proba[1] for proba in y_probas]
        cel = log_loss(y_true, y_probas_pos)  # cross-entropy loss
        print(f"log loss: {cel}")
        roc_auc = roc_auc_score(y_true, y_probas_pos)
        print(f"roc auc: {roc_auc}")

    return (accuracy, precision, recall, fbeta)


def plot_charts(
                y_true: list[int] | np.ndarray,
                y_pred: list[int] | np.ndarray,
                y_probas: list[list[float]] | np.ndarray,
            ) -> None:
    """Plot charts from predictions.

    Currently, this function plots charts with :class:`ConfusionMatrixDisplay`,
    :class:`PrecisionRecallDisplay`, and :class:`RocCurveDisplay`.
    """
    # evaluate with more statistics
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.show()

    # Precision Recall curve
    pr_display = PrecisionRecallDisplay.from_predictions(
                    y_true=y_true, y_pred=y_pred)
    pr_display.plot()
    plt.show()

    # Roc curve
    RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_probas)


def split_list(lis: list, parts: int = 5) -> list[list]:
    """Split a list into sub-arrays of even sizes.

    If the length of the original list is not divisible by `parts`,
    the lengths of resulting subarrays may variate within the range of
    ``(len(list)/parts)+parts`` < result < ``(len(list)/parts)+parts``.

    .. note::
        This function should be able to operate on np.ndarrays, but use
        :func:`np.split` for that purpose.

    Returns:
        list of split sub-arrays.

    Raises:
        ValueError: if the length of the original list is shorter than
            the requested number of partitions.
    """
    if len(lis) % parts != 0:
        print(
            f"NOTE: the number of training samples ({len(lis)}) "
            + f"is not divisible by {parts}."
        )
        print("Resulting split of sub-arrays will be uneven.")

    if len(lis) < parts:
        msg = f"Cannot divide a list of length {len(lis)} into {parts} parts!"
        raise ValueError(msg)

    # Calculate where to split the array
    split_ids = []
    separator = len(lis) / parts
    idx = separator

    while idx < len(lis):
        split_ids.append(int(idx))
        idx += separator

    results = []
    split_start = 0
    split_end = split_ids[0] - 1
    for i in split_ids:
        results.append(lis[split_start : int(split_end)])
        split_start = i - 1
        split_end += separator
    results.append(lis[split_start:])
    return results


def evaluate_classifier(
    clf: BoWEstimator,
    loaded_ds: LoadedDataset,
    *,
    plot: bool = False,
    threshold: float = 0.5
    ) -> tuple[
            ProbaPredictions, ProbaPredictions,
            Mislabels, Mislabels]:
    """Evaluate a classifier with provided test sets.

    Args:
        clf: a :class:`src.classifier.wrappers.BoWEstimator` instance.
        loaded_ds: a :class:`src.classifier.dataset_skeleton.LoadedDataset` instance
            for the dataset to train and evaluate the classifier with.
        plot: if True, create charts
            and display them with :func:`src.classifier.eval_utils.plot_charts`.
        threshold: the threshold of predicted probability at which to decide
            that a sample should be classified as belonging to
            a particular class.

    Returns:
        :class:`ProbaPredictions` instances, one for OT and another for NT,
        along with :class:`Mislabels` instances for OT and NT.
    """
    ot_test_x = loaded_ds.test.get_samples(0)
    nt_test_x = loaded_ds.test.get_samples(1)
    # cast / convert lists from the dataset to np.ndarray
    ot_test_y = np.array(loaded_ds.test.get_labels(0), dtype=np.int64)
    nt_test_y = np.array(loaded_ds.test.get_labels(1), dtype=np.int64)
    all_test_y = np.array(loaded_ds.test.get_labels(), dtype=np.int64)
    print(f"test size: {all_test_y.shape}")

    # predict probabilities with clf
    print("OT --->")
    ot_proba_preds = predict_proba(
        clf,
        ot_test_x,
        threshold=threshold
    )
    ot_mislabels = mislabel_stats(
                ot_test_x,
                np.array(ot_test_y),
                ot_proba_preds
            )
    print(f"Total probas: {ot_proba_preds.get_total_probas()}")
    print("NT --->")
    nt_proba_preds = predict_proba(
        clf,
        nt_test_x,
        threshold=threshold
    )
    nt_mislabels = mislabel_stats(
                nt_test_x,
                np.array(nt_test_y),
                nt_proba_preds
            )
    print(f"Total probas: {nt_proba_preds.get_total_probas()}")
    print("All --->")
    pred_y_all = ot_proba_preds.predictions
    pred_y_all = np.append(pred_y_all, nt_proba_preds.predictions, axis=0)
    # print(f"len y all: {len(pred_y_all)} ~ len ot {len(ot_proba_preds.predictions)} len nt {len(nt_proba_preds.predictions)}")
    probas = ot_proba_preds.get_probas()
    probas = np.append(probas, nt_proba_preds.get_probas(), axis=0)
    assert (len(ot_proba_preds.predictions) == len(ot_test_x))
    metricise(all_test_y, y_all=pred_y_all, y_probas=probas)

    if plot:
        plot_charts(all_test_y, pred_y_all, probas)

    return (ot_proba_preds, nt_proba_preds, ot_mislabels, nt_mislabels)


def eval_and_save(  # noqa: PLR0913
        clf: BoWEstimator,
        loaded: LoadedDataset,
        file_formatter: Callable,
        *,
        out_dir: str = "./out/",
        save_file_prefix: str = "",
        save_file_suffix: str = "",
        save_file_ext: str = ".csv",
        threshold: float = 0.5,
    ) -> tuple[BoWEstimator, list[ProbaPredictions]]:
    """Wrapper around evaluate_classifier, save_mislabels, and save_all_preds.

    Args:
        clf: a classifier wrapped in :class:`src.classifier.wrappers.BoWEstimator`
        loaded: a dataset loaded from files as a
            :class:`src.classifier.dataset_skeleton.LoadedDataset` instance.
        file_formatter: any callable object (function, method, etc.)
            that returns a formatted string which can be directly
            written to a file.
        out_dir: Path or string of path to the directory to save result files.
        save_file_prefix: Prefixes to add before/after the default file name
            for each evaluation process. These are used to construct save file
            names passed to save_mislabels and save_all_preds functions.
        save_file_suffix: Suffix in filenames. **Note** this is different
            from the *file extension* defined with param ``save_file_ext``.
            See also the param ``save_file_prefix``.
        save_file_ext: File extension for the save file. By default, it's CSV.
        threshold: the threshold of probability to classify a certain sample
            as belonging to a particular class.

    Returns:
        A BoWEstimator instance and two ProbaPredictions, each for OT and NT.

    .. note::
        This function uses :meth:`pathlib.Path.write_text` directly.
        Error descriptions in this documentation are not thorough.

    .. seealso::
        :func:`src.classifier.eval_utils.evaluate_classifier`
            for params: ``clf``, ``ot_test_X``, ``ot_test_y``,
            ``nt_test_X``, ``nt_test_y``
        :func:`src.classifier.eval_utils.save_mislabels`
        :func:`src.classifier.eval_utils.save_all_preds`
            for param: ``file_formatter``.
    """
    # convert the out_dir to Path
    out_dir_p = Path(out_dir)
    # Remove prepended slash in save_file_prefix
    # since they break Path concatenation
    if save_file_prefix[0] == "/":
        save_file_prefix = save_file_prefix[1:]

    # evaluate the classifier with the provided samples
    (ot_probas, nt_probas, ot_mislabels, nt_mislabels) = evaluate_classifier(
        clf, loaded, threshold=threshold
    )

    # Construct save files' paths
    ot_mislabels_file = Path(
        save_file_prefix
        + "prediction_mislabels_ot"
        + save_file_suffix
        + save_file_ext
    )
    nt_mislabels_file = Path(
        save_file_prefix
        + "prediction_mislabels_nt"
        + save_file_suffix
        + save_file_ext
    )
    ot_all_file = Path(
        save_file_prefix
        + "prediction_all_ot"
        + save_file_suffix
        + save_file_ext
    )
    nt_all_file = Path(
        save_file_prefix
        + "prediction_all_nt"
        + save_file_suffix
        + save_file_ext
    )

    # Save the evaluation results to files
    ot_mislabels.save_to_file(file_formatter, (out_dir_p / ot_mislabels_file))
    nt_mislabels.save_to_file(file_formatter, (out_dir_p / nt_mislabels_file))

    ot_probas.save_to_file(file_formatter, (out_dir_p / ot_all_file))
    nt_probas.save_to_file(file_formatter, (out_dir_p / nt_all_file))

    return (clf, [ot_probas, nt_probas])


def cross_validate(
        clf: BoWEstimator,
        training_x: list[Verse],
        training_y: list[int],
        fold: int = 5,
        threshold: float = 0.5
    ) -> None:
    """Perform cross validation with the provided training set.

    .. attention:: cross validation should be performed with the training set,
        and you still need to hold out the test set for final evaluation.
    """
    print("> Cross-Validation <")
    skf_splitter = StratifiedKFold(n_splits=fold)
    splits = skf_splitter.split(training_x, training_y)
    accs = []
    precs = []
    recs = []
    fones = []

    for train_g, test_g, in splits:
        proba_c = BoWEstimator(clone(clf.algo), clf.n_gram_formatter, clf.n)
        train_ids = list(train_g)
        test_ids = list(test_g)
        # train_samples = train[0]
        # train_labels = train[1]
        train_verses = [training_x[int(idx)] for idx in train_ids]
        train_labels = [training_y[int(idx)] for idx in train_ids]
        test_verses = [training_x[int(idx)] for idx in test_ids]
        test_labels = [training_y[int(idx)] for idx in test_ids]
        # test_samples = test[0]
        # test_labels = test[1]

        proba_c.fit(train_verses, train_labels)
        predictions = predict_proba(proba_c,
                                            test_verses,
                                            threshold=threshold
                                    )
        acc, prec, rec, fone = metricise(test_labels,
                                         y_all=predictions.predictions)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        fones.append(fone)
    print(f"Accuracy > avg: {np.average(accs)}, std: {np.std(accs)}")
    print(f"Precision > avg: {np.average(precs)}, std: {np.std(precs)}")
    print(f"Recall > avg: {np.average(recs)}, std: {np.std(recs)}")
    print(f"F1 score > avg: {np.average(fones)}, std: {np.std(fones)}")
