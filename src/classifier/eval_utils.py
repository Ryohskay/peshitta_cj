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
import logging
from pathlib import Path
from typing import Literal, TypedDict

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

logger = logging.getLogger(__name__)


def mislabel_stats(
    preds: ProbaPredictions,
) -> Mislabels | None:
    """Calculate and print some basic statistics on model inputs & outputs.

    Args:
        preds: a :class:`src.classifier.result_utils.Predictions` instance for
            the results of predictions on the input. When a
            :class:`src.classifier.result_utils.ProbaPredictions` instance is
            passed, the returned ``Mislabels`` class instance will have
            the ``probas`` attribute set.

    Returns:
        an instance of :class:`src.classifier.result_utils.Mislabels`. `None`
        if no verse was mislabelled.

    Raises:
        ValueError: if the attribute ``preds.correct_labels`` is ``None``.
    """
    if preds.correct_labels is None:
        msg = (
            "The attr `preds.correct_labels` is empty. Correct labels are "
            + "required to find mislabelled verses."
        )
        raise ValueError(msg)

    # print some stats about the whole dataset
    print(
        "Local accuracy: "
        + f"{accuracy_score(preds.correct_labels, preds.predictions):.04f}"
    )

    mislab_books = preds.find_book_mislabels()
    # Flatten the per-book mislabelled results into a Mislabels class instance
    mislabelled_preds = []
    y_correct = []
    mislabelled_verses = []
    mislabel_probas = []

    for book in mislab_books:
        mislabelled_preds.extend(book.predictions)
        y_correct.extend(book.correct_labels)
        mislabelled_verses.extend(book.mislabels)
        mislabel_probas.extend(book.probas)

    if len(mislabelled_preds) > 0:
        mislabs = Mislabels(
            mislabelled_preds,
            y_correct,
            mislabelled_verses,
            mislabel_probas,
        )

        # Print some stats about mislabelled verses
        print(
            f"Number of mislabelled points out of the total {len(preds.samples)} "
            + f"verses: {len(mislabs)} "
            + f"({mislabelled_preds.count(-1)} labelled as unknown)"
        )

        return mislabs
    return None


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


def plot_charts(  # noqa: PLR0913
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_probas: list[list[float]] | np.ndarray,
    pos_label: int = 1,
    *,
    save_fname: SavefileName | None = None,
    out_dir: str | Path | None = Path("./graphics/out/"),
) -> None:
    """Plot charts from predictions.

    Args:
        y_true: correct labels
        y_pred: predicted labels
        y_probas: predicted probabilities
        pos_label: positive label for the ROC curve
        save_fname: a :class:`src.classifier.fname_utils.SavefileName` instance
            containing the file name for a classifier.
        out_dir: Path or string of path to the directory to save result files.
            If ``save_fname`` is not None, the charts will be saved to this
            directory with the name of ``save_fname``. If the directory name
            is not provided, the charts will be plotted
            on a graphical interface.
    """
    save_fname_p = None
    if save_fname is not None and out_dir is not None:
        save_fname.ext = "jpeg"
        save_fname_p = Path(out_dir) / save_fname.get_fname()

    # evaluate with more statistics
    cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    cm_display.plot()
    save_dest = (
        save_fname_p.parent / save_fname_p.rename(save_fname_p.stem + "_cm" + save_fname_p.suffix)
        if save_fname_p is not None
        else f"{out_dir}/confusion_matrix.jpeg"
    )
    if out_dir is None:
        plt.show()
    else:
        plt.savefig(save_dest, bbox_inches="tight")
    plt.close("all")

    # Precision Recall curve
    pr_display = PrecisionRecallDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred
    )
    pr_display.plot()
    save_dest = (
        save_fname_p.parent / save_fname_p.rename(save_fname_p.stem + "_pr" + save_fname_p.suffix)
        if save_fname_p is not None
        else f"{out_dir}/precision_recall.jpeg"
    )
    if out_dir is None:
        plt.show()
    else:
        plt.savefig(save_dest, bbox_inches="tight")
    plt.close("all")

    # Roc curve
    proba_pred_pos = [proba[pos_label] for proba in y_probas]
    roc_display = RocCurveDisplay.from_predictions(
        y_true=y_true, y_pred=proba_pred_pos
    )
    roc_display.plot()
    save_dest = (
        save_fname_p.parent / save_fname_p.rename(save_fname_p.stem + "_roc" + save_fname_p.suffix)
        if save_fname_p is not None
        else f"{out_dir}/roc_auc.jpeg"
    )
    if out_dir is None:
        plt.show()
    else:
        plt.savefig(save_dest, bbox_inches="tight")
    plt.close("all")


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
        msg = (
            f"NOTE: the number of training samples ({len(lis)}) "
            + f"is not divisible by {parts}. "
            + "Resulting split of sub-arrays will be uneven."
        )
        logger.info(msg)

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
    split_end = split_ids[0]
    for i in split_ids:
        results.append(lis[split_start : int(split_end)])
        split_start = i
        split_end += separator
    results.append(lis[split_start:])
    return results


class SummaryDict(TypedDict):
    """TypedDict for the summary dictionary.

    Attributes:
        n_gram_form: the n-gram form of the classifier.
        n: the n-gram size of the classifier.
        total_n_grams_parsed: the total number of n-grams parsed.
        top_ten_in_training
    """

    n_gram_form: Literal["word", "char"]
    n: int
    total_n_grams_parsed: int
    top_ten_in_training: list[tuple[tuple[str], int]]
    test_mislabel_percent: dict[str, float]
    metrics: ResultStats


def get_summary(
    clf: BoWEstimator,
    test_split: DataSplit,
    ot_mislab: Mislabels,
    nt_mislab: Mislabels,
    res_stats: ResultStats,
) -> SummaryDict:
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
    # percentage of mislabelled verses out of all supports per each class.
    if ot_mislab is not None:
        ot_mislab_propo = (len(ot_mislab.mislabels)
                           / len(test_split.get_labels(0)))
    else:
        ot_mislab_propo = 0.0

    if nt_mislab is not None:
        nt_mislab_propo = (len(nt_mislab.mislabels)
                           / len(test_split.get_labels(1)))
    else:
        nt_mislab_propo = 0.0

    mislab_percents = {
        label_data.ValToLabel[0]: ot_mislab_propo * 100,
        label_data.ValToLabel[1]: nt_mislab_propo * 100,
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
    plot: bool = True,
    threshold: float = 0.5,
) -> tuple[
    ProbaPredictions, ProbaPredictions, Mislabels | None, Mislabels | None, ResultStats
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
        ``ResultStats`` instance. The `Mislabels` instances may be `None` if
        no mislabelled verses for the corresponding class were found.

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
    ot_mislabels = mislabel_stats(ot_proba_preds)
    print("NT --->")
    nt_proba_preds = predict_proba(
        clf, nt_test_x, nt_test_y, threshold=threshold
    )
    nt_mislabels = mislabel_stats(nt_proba_preds)
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
    measurements.append(
        ThresholdStats(
            target_thresholds[0], acc, list(prc), list(rec), list(f1)
        )
    )

    # measure scores at other thresholds
    for thresh in target_thresholds[1:]:
        pred_y_all = convert(probas, thresh)
        acc, prc, rec, f1, _ = metricise(all_test_y, pred_y_all, do_print=False)
        measurements.append(
            ThresholdStats(thresh, acc, list(prc), list(rec), list(f1))
        )

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
    do_plot: bool = True,
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

    # Save the prediction results to files
    ot_probas.save_to_file(
        file_formatter, (out_dir_p / ot_all_file.get_fname())
    )
    nt_probas.save_to_file(
        file_formatter, (out_dir_p / nt_all_file.get_fname())
    )

    # save the mislabelled verses to files
    if ot_mislabels is not None:
        ot_mislabels_file = ot_all_file.copy()
        ot_mislabels_file.mark_special_file(is_mislabel=True)
        ot_mislabels.save_to_file(
            file_formatter, (out_dir_p / ot_mislabels_file.get_fname())
        )

    if nt_mislabels is not None:
        nt_mislabels_file = nt_all_file.copy()
        nt_mislabels_file.mark_special_file(is_mislabel=True)
        nt_mislabels.save_to_file(
            file_formatter, (out_dir_p / nt_mislabels_file.get_fname())
        )

    # save per-book total probabilities
    save_total_proba_fname = save_fname.copy()
    save_total_proba_fname.mark_special_file(is_total_proba=True)
    total_proba_save_fp = out_dir_p / save_total_proba_fname.get_fname()
    total_proba_csv = csvify_total_proba(ot_probas.get_total_probas())
    total_proba_csv += csvify_total_proba(
        nt_probas.get_total_probas(), no_header=True
    )
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
) -> tuple[list, list, list, list]:
    """Perform cross validation with the provided training set.

    .. attention:: cross validation should be performed with the training set,
        and you still need to hold out the test set for final evaluation.

    Returns:
        a tuple of lists containing accuracy, precision, recall, and f1 score
        for each fold.

    Raises:
        ValueError: if the length of the training set is shorter than
            the number of requested folds.
    """
    print("> Cross-Validation <")
    skf_splitter = StratifiedKFold(n_splits=fold)
    if len(training_x) < fold:
        msg = (
            f"Cannot divide a list of length {len(training_x)} into {fold} parts!"
        )
        raise ValueError(msg)
    splits = skf_splitter.split(training_x, training_y)  # type: ignore[reportArgumentType]
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
            clf.n,
        )
        proba_c.preprocessor = clf.preprocessor
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
    return accs, precs, recs, f_ones
