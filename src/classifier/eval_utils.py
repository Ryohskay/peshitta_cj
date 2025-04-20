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

import json
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

from src.classifier.dataset_skeleton import DataSplit, LoadedDataset
from src.classifier.fitting_utils import identity
from src.classifier.fname_utils import SavefileName
from src.classifier.prediction_utils import convert, predict_proba
from src.classifier.result_utils import (
    FileFormatterProto,
    Mislabels,
    ProbaPredictions,
    ResultStats,
    ThresholdStats,
    Verse,
    jsonify_dict,
)
from src.classifier.wrappers import BoWEstimator
from src.shared import label_data


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
        msg = (
            "Cannot measure the size of `inputs`!"
            + " It seems like the argument `inputs` is empty."
        )
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
        if current_book != preds.samples[i].book:
            # if current_book is not empty,
            # calculate some statistics and print
            if current_book:
                num_mislabels = len(mislabel_books[current_book])
                print(
                    f">> {current_book}: {num_mislabels} "
                    + "mislabelled verses, accounting for "
                    + f"{num_mislabels / num_book_verses:.02f}% out of "
                    + f"{num_book_verses} verses ({num_book_unk} verses "
                    + "labelled unknown)"
                )
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
                mislabel_books.update({current_book: [preds.samples[i]]})
            else:
                mislabel_books[current_book].append(preds.samples[i])
            if preds.predictions[i] == -1:
                num_book_unk += 1

    if current_book:
        num_mislabels = len(mislabel_books[current_book])
        print(
            f">> {current_book}: {num_mislabels} "
            + "mislabelled verses, accounting for "
            + f"{(num_mislabels / num_book_verses) * 100:.02f}% out of "
            + f"{num_book_verses} verses ({num_book_unk} verses "
            + "labelled unknown)"
        )

    # Collate the collected mislabelling results into a Mislabels class instance
    mislabs = Mislabels(
        incorrect_labels,
        correct_labels,
        [mislab for book in mislabel_books for mislab in mislabel_books[book]],
        mislab_probas,
    )

    # Print some stats
    print(
        f"Number of mislabelled points out of the total {sample_size} "
        + f"verses: {len(mislabs)} "
        + f"({incorrect_labels.count(-1)} labelled as unknown)"
    )
    print(
        "Local accuracy: "
        + f"{accuracy_score(y_correct, preds.predictions):.04f}"
    )

    return mislabs


def metricise(
    y_true: list[int] | NDArray,
    y_all: list[int] | NDArray,
    y_probas: list[list[float]] | NDArray | None = None,
    *,
    do_print: bool = True,
) -> tuple[float, NDArray, NDArray, NDArray, ResultStats | None]:
    """Calculate performance metrics of a classifier using the outputs.

    Args:
        y_true: Correct (gold reference) labels
        y_all: all predictions in a one-dimensional list.
        y_probas: probabilities predicted for each sample.
        do_print: if True, print the results of the evaluation.

    Returns:
        accuracy (value), precision (1D array), recall (1D array),
        and f beta scores (1D array) as floats. If the ``y_probas`` argument is
        provided, also returns a :class:`ResultStats` instance.

    Raises:
        ValueError: if the length of ``y_all`` after :func:`numpy.array` and
            the length of ``y_true`` do not match
    """
    # type annotations fail in this function due to very loose typing in
    # the scikit-learn library. They are explicitly ignored by the special
    # inline comments for pyright.
    y_pred = np.array(y_all)

    if len(y_pred) != len(y_true):
        msg = f"y_pred has length of {len(y_pred)}, but y_true {len(y_true)}"
        raise ValueError(msg)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, fbeta, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )

    if do_print:
        print(f"Overall Accuracy: {accuracy:.04f}")
        print(f"Precisions: [0] {precision[0]:.04f}, [1] {precision[1]:.04f}")  # type: ignore[reportIndexIssue]
        print(f"Recalls: [0] {recall[0]:.04f}, [1] {recall[1]:.04f}")  # type: ignore[reportIndexIssue]
        print(f"F1 Score: [0] {fbeta[0]:.04f}, [1] {fbeta[1]:.04f}")  # type: ignore[reportIndexIssue]
        print(f"Supports: [0] {support[0]:.04f}, [1] {support[1]:.04f}")  # type: ignore[reportIndexIssue]

    if y_probas is None:
        # if probabilities are not provided, finish calculations here
        return (accuracy, precision, recall, fbeta, None)  # type: ignore[reportReturnType]

    # if y_probas is not None and
    if len(y_probas) != len(y_true):
        msg = (
            f"length of y_probas {len(y_probas)} is not equal to "
            + f"length of y_true {len(y_true)}"
        )
        raise ValueError(msg)

    # if y_probas is not None:
    # calculate stats that use probabilities
    y_probas_zero = [proba[0] for proba in y_probas]
    y_probas_one = [proba[1] for proba in y_probas]

    # cross-entropy loss
    cel_res = [log_loss(y_true, y_probas_zero), log_loss(y_true, y_probas_one)]
    # area under curve
    roc_auc_res = [
        roc_auc_score(y_true, y_probas_zero),
        roc_auc_score(y_true, y_probas_one),
    ]
    # print probability-based stats
    if do_print:
        print(f"log loss: {cel_res}")
        print(f"roc auc: {roc_auc_res}")
    stats = ResultStats(support, cel_res, roc_auc_res)  # type: ignore[reportReturnType]
    return (accuracy, precision, recall, fbeta, stats)  # type: ignore[reportReturnType]


def csvify_total_proba(
    total_proba_dict: dict[str, list[float]], *, no_header: bool = False
) -> str:
    """Format the total proba dict data into CSV.

    Args:
        total_proba_dict: a dictionary where each entry records the total
            probability of a book belonging to a particular class.
        no_header: a boolean indicating if the returned str should contain
            CSV column headers. Setting this to ``True`` is useful when
            concatenating multiple total proba dicts into one CSV file.

    Returns:
        the total probability of books as a CSV-formatted str.
    """
    if no_header:
        total_proba_csv = ""
    else:
        total_proba_csv = (
            "Book,"
            + f"Probability for {label_data.ValToLabel[0]},"
            + f"Probability for {label_data.ValToLabel[1]}\n"
        )

    print(">> per-book total probabilities:")
    for prod_book in total_proba_dict:
        print(
            f"{prod_book}: (OT) {total_proba_dict[prod_book][0]:.04f}, "
            + f"(NT) {total_proba_dict[prod_book][1]:.04f}"
        )
        total_proba_csv += (
            f"{prod_book},{total_proba_dict[prod_book][0]},"
            + f"{total_proba_dict[prod_book][1]}\n"
        )
    return total_proba_csv


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
        y_true=y_true, y_pred=y_pred
    )
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


def get_summary(
    clf: BoWEstimator,
    test_split: DataSplit,
    ot_mislab: Mislabels,
    nt_mislab: Mislabels,
    res_stats: ResultStats,
) -> dict:
    """Make a summary of the classifier evaluation results.

    Args:
        clf: a :class:`src.classifier.wrappers.BoWEstimator` instance;
            the classifier must be trained before calling this function.
        test_split: a :class:`src.classifier.dataset_skeleton.DataSplit`
            instance for the test set to evaluate the classifier with.
        ot_mislab: a :class:`src.classifier.result_utils.Mislabels` instance
            for the OT verses.
        nt_mislab: a :class:`src.classifier.result_utils.Mislabels` instance
            for the NT verses.
        res_stats: a :class:`src.classifier.result_utils.ResultStats` instance
            containing the evaluation results of the classifier.

    Raises:
        ValueError: if the classifier has not been trained.
    """
    if clf.vocabs is None:
        msg = "The classifier has not been fit yet!"
        raise ValueError(msg)

    # test if the classifier is a word n-gram classifier by checking the
    # n-gram formatter.
    n_gram_form = "word" if clf.n_gram_formatter == identity else "char"
    mislab_percents = {
        label_data.ValToLabel[0]: (
            len(ot_mislab.mislabels) / len(test_split.get_labels(0))
        )
        * 100,
        label_data.ValToLabel[1]: (
            len(nt_mislab.mislabels) / len(test_split.get_labels(1))
        )
        * 100,
    }
    return {
        "n_gram_form": n_gram_form,
        "n": clf.n,
        "total_n_grams_parsed": clf.vocabs[0].total(),
        "top_ten_in_training": clf.vocabs[0].most_common(10),
        "test_mislabel_percent": mislab_percents,
        "metrics": res_stats,
    }


def evaluate_classifier(
    clf: BoWEstimator,
    test_ds: DataSplit,
    *,
    plot: bool = False,
    threshold: float = 0.5,
) -> tuple[
    ProbaPredictions, ProbaPredictions, Mislabels, Mislabels, ResultStats
]:
    """Evaluate a classifier with provided test sets.

    Args:
        clf: a :class:`src.classifier.wrappers.BoWEstimator` instance, which has
            already been fit.
        test_ds: a :class:`src.classifier.dataset_skeleton.DataSplit` instance
            for the test set to evaluate the classifier with.
        plot: if True, create charts
            and display them with :func:`src.classifier.eval_utils.plot_charts`.
        threshold: the threshold of predicted probability at which to decide
            that a sample should be classified as belonging to
            a particular class.

    Returns:
        :class:`ProbaPredictions` instances, one for OT and another for NT,
        :class:`Mislabels` instances for OT and NT, as well as a
        ``ResultStats`` instance.

    Raises:
        RuntimeError: if the result of `metricise` function fails to return
            a `ResultStats` instance even if probabilities are provided.
    """
    ot_test_x = test_ds.get_samples(0)
    nt_test_x = test_ds.get_samples(1)
    # cast / convert lists from the dataset to np.ndarray
    ot_test_y = np.array(test_ds.get_labels(0), dtype=np.int64)
    nt_test_y = np.array(test_ds.get_labels(1), dtype=np.int64)
    all_test_y = np.array(test_ds.get_labels(), dtype=np.int64)
    print(f"test size: {all_test_y.shape}")

    # predict probabilities with clf
    print("OT --->")
    ot_proba_preds = predict_proba(
        clf, ot_test_x, ot_test_y, threshold=threshold
    )
    ot_mislabels = mislabel_stats(ot_test_x, ot_test_y, ot_proba_preds)
    print("NT --->")
    nt_proba_preds = predict_proba(
        clf, nt_test_x, nt_test_y, threshold=threshold
    )
    nt_mislabels = mislabel_stats(
        nt_test_x, np.array(nt_test_y), nt_proba_preds
    )
    print("All --->")
    probas = ot_proba_preds.get_probas()
    probas = np.append(probas, nt_proba_preds.get_probas(), axis=0)

    # define thresholds to measure scores
    target_thresholds = [0.5, 0.8, 0.9, 0.95]
    if threshold not in target_thresholds:
        target_thresholds.append(threshold)
        target_thresholds.sort()

    measurements = []

    # calculate scores at the first threshold
    pred_y_all = convert(probas, target_thresholds[0])
    print(f"> with threshold: {target_thresholds[0]}")
    acc, prc, rec, f1, stats = metricise(all_test_y, pred_y_all, probas)
    # verify that the stats are not None
    if stats is None:
        msg = "The result of `metricise` function with probabilities was None!"
        raise RuntimeError(msg)
    # add the calculated stats to the measurements
    measurements.append(ThresholdStats(
        target_thresholds[0], acc, list(prc), list(rec), list(f1)))

    # measure scores at other thresholds
    for thresh in target_thresholds[1:]:
        pred_y_all = convert(probas, thresh)
        acc, prc, rec, f1, _ = metricise(all_test_y, pred_y_all, do_print=False)
        measurements.append(ThresholdStats(
            thresh, acc, list(prc), list(rec), list(f1)))

    # register the calculated measurements
    stats.add_thresh_stats(measurements)

    if plot:
        plot_charts(all_test_y, pred_y_all, probas)

    return (ot_proba_preds, nt_proba_preds, ot_mislabels, nt_mislabels, stats)


def eval_and_save(  # noqa: PLR0913
    clf: BoWEstimator,
    loaded: LoadedDataset,
    file_formatter: FileFormatterProto,
    save_fname: SavefileName,
    *,
    out_dir: str = "./out/",
    do_plot: bool = False,
    threshold: float = 0.5,
) -> tuple[BoWEstimator, list[ProbaPredictions]]:
    """Wrapper around evaluate_classifier, save_mislabels, and save_all_preds.

    Args:
        clf: a classifier wrapped in
            :class:`src.classifier.wrappers.BoWEstimator`, which has already
            been fit.
        loaded: a dataset loaded from files as a
            :class:`src.classifier.dataset_skeleton.LoadedDataset` instance.
        file_formatter: any callable object (function, method, etc.)
            that returns a formatted string which can be directly
            written to a file.
        save_fname: :class:`src.classifier.fname_utils.SavefileName` instance
            containing the base file name information for this classifier.
        out_dir: Path or string of path to the directory to save result files.
        do_plot: if ``eval_classifier`` should plot diagrams using matplotlib.
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

    # evaluate the classifier with the provided samples
    (ot_probas, nt_probas, ot_mislabels, nt_mislabels, results) = (
        evaluate_classifier(clf, loaded.test, threshold=threshold, plot=do_plot)
    )

    # Configure and prepare save files' names
    ot_all_file = save_fname.copy()
    ot_all_file.set_scope(label_data.ValToLabel[0])
    nt_all_file = save_fname.copy()
    nt_all_file.set_scope(label_data.ValToLabel[1])

    ot_mislabels_file = ot_all_file.copy()
    ot_mislabels_file.mark_special_file(is_mislabel=True)
    nt_mislabels_file = nt_all_file.copy()
    nt_mislabels_file.mark_special_file(is_mislabel=True)

    # Save the prediction results to files
    ot_mislabels.save_to_file(
        file_formatter, (out_dir_p / ot_mislabels_file.get_fname())
    )
    nt_mislabels.save_to_file(
        file_formatter, (out_dir_p / nt_mislabels_file.get_fname())
    )

    ot_probas.save_to_file(
        file_formatter, (out_dir_p / ot_all_file.get_fname())
    )
    nt_probas.save_to_file(
        file_formatter, (out_dir_p / nt_all_file.get_fname())
    )

    # save per-book total probabilities
    save_total_proba_fname = save_fname.copy()
    save_total_proba_fname.mark_special_file(is_total_proba=True)
    total_proba_save_fp = out_dir_p / save_total_proba_fname.get_fname()
    total_proba_csv = csvify_total_proba(ot_probas.get_total_probas())
    total_proba_csv += csvify_total_proba(nt_probas.get_total_probas())
    total_proba_save_fp.write_text(total_proba_csv)

    # create a summary dictionary of a classifier evaluation
    summary = get_summary(clf, loaded.test, ot_mislabels, nt_mislabels, results)
    # save the classifier summary to a file
    summary_fname = save_fname.copy()
    # change file type to json for easier handling upon UI integration
    summary_fname.ext = "json"
    summary_fname.mark_special_file(is_clf_summary=True)
    summary_fp = out_dir_p / summary_fname.get_fname()
    with summary_fp.open("w") as sfp:
        json.dump(summary, sfp, default=jsonify_dict)

    return (clf, [ot_probas, nt_probas])


def cross_validate(
    clf: BoWEstimator,
    training_x: list[Verse],
    training_y: list[int],
    fold: int = 5,
    threshold: float = 0.5,
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
    f_ones = []

    for (
        train_g,
        test_g,
    ) in splits:
        proba_c = BoWEstimator(
            clone(clf.algo),  # type: ignore[reportArgumentType]
            clf.n_gram_formatter,
            clf.n
        )
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
        predictions = predict_proba(
            proba_c, test_verses, test_labels, threshold=threshold
        )
        acc, prec, rec, fone, _ = metricise(
            test_labels, y_all=predictions.predictions
        )
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f_ones.append(fone)
    print(f"Accuracy > avg: {np.average(accs)}, std: {np.std(accs)}")
    print(f"Precision > avg: {np.average(precs)}, std: {np.std(precs)}")
    print(f"Recall > avg: {np.average(recs)}, std: {np.std(recs)}")
    print(f"F1 score > avg: {np.average(f_ones)}, std: {np.std(f_ones)}")
