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

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
    accuracy_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from classifier.result_utils import (
    Mislabels,
    Predictions,
    ProbaPredictions,
    Verse,
)
from classifier.wrappers import (
    BoWEstimator,
    PredictorProto,
)


def quick_stats(
    inputs: np.ndarray | list[Any],
    y_correct: np.ndarray,
    prediction: np.ndarray,
) -> tuple[int, int]:
    """Calculate and print some basic statistics on model inputs & outputs.

    Args:
        inputs: Inputs passed when the classifier
            produced the predictions.
        y_correct: Correct (gold-standard) labels
            for the input verses.
        prediction: Labels for the input verses
            predicted by the classifier.

    Returns:
        number of mislabelled verses and that of
        correctly labelled verses.

    Raises:
        ValueError: If the provided list of inputs to test is either zero
            length or does not have a measurable length.
    """
    sample_size = 0
    if hasattr(inputs, "shape") and callable(inputs.shape):  # type: ignore[reportAttributeAccessIssue]
        # AttributeAccessIssue can be ignored since AttributeError is explicitly
        # handled
        sample_size = int(inputs.shape[0])  # type: ignore[reportAttributeAccessIssue]
    else:
        # If inputs is not a np.ndarray,
        # use the standard len() function instead.
        sample_size = len(inputs)

    if sample_size == 0:
        msg = "Cannot measure the size of model input array X!"
        raise ValueError(msg)

    num_mislabels = (y_correct != prediction).sum()
    num_correct = sample_size - num_mislabels

    print(
        f"Number of mislabelled points out of the total {sample_size} "
                + f"verses: {num_mislabels}"
    )
    print(f"Local accuracy: {accuracy_score(y_correct, prediction):.02f}")

    return (num_mislabels, num_correct)


def find_mislabels(
        y_correct: list | np.ndarray,
        y_pred: list | np.ndarray,
        test_x: list[Verse],
        probas: np.ndarray
    ) -> Mislabels:
    """Find and return mislabelled verses in string format.

    Args:
        y_correct: list of correct labels
        y_pred: list of predictions
        test_x: list of verses corresponding to the labels
        probas: list of probabilities, of size (numbert of samples,
            number of prediction classes)

    Returns:
        A :class:`Mislabels` instance.
    """
    # initialise storages for mislabelled verse's info
    mislabel_idcs = np.empty(0, dtype=np.int_)
    mislabels = np.empty(0, dtype=np.int_)
    correct_labels = np.empty(0, dtype=np.int_)

    # parse through all the predictions,
    # find and record occasions where they don't match ``y_correct``
    for i in range(len(y_correct)):
        if y_correct[i] != y_pred[i]:
            mislabel_idcs = np.append(mislabel_idcs, i)
            mislabels = np.append(mislabels, y_pred[i])
            correct_labels = np.append(
                    correct_labels,
                    y_correct[i]
                    )

    results = Mislabels(
                            mislabel_idcs,
                            mislabels,
                            correct_labels
                        )

    if len(results) > 0:
        # extract list verses at idcs and set results.verses
        results.extract_verses(test_x)

    if probas is not None:
        results.probas = probas[results.idcs]

    return results


def predict(
        classifier: PredictorProto,
        test_samples: np.ndarray | list,
        test_labels: np.ndarray
    ) -> Predictions:
    """Predict on the data with a classifier and get some simple statistics.

    Args:
        classifier: any object that has a method `.predict()`.
        test_samples: inputs to the classifier
        test_labels: correct labels (gold references) for the inputs. This is
            not used to predict probabilities, but is used to calculate
            prediction accuracy etc.

    Returns:
        A :class:`Prediction` class instance.
    """
    y_pred = classifier.predict(test_samples)
    num_mislabels, num_correct = quick_stats(
                                            np.array(test_samples),
                                            np.array(test_labels),
                                            np.array(y_pred)
                                        )
    return Predictions(test_samples, y_pred,
                test_labels, num_correct, num_mislabels)


def _convert(probas: np.ndarray, thresh: float) -> np.ndarray:
    """Convert list of probabilities to a list of labels.

    Args:
        probas: list of predicted probabilities.
        thresh: threshold to decide if a probability prediction should be
            labelled as an instance of the class.

    Returns:
        0 if prediction for label 0 is over the threshold,
        1 if prediction for label 1 is over the threshold,
        -1 if probabilities for both labels do not exceed the threshold.

    Raises:
        ValueError: if neither of the classes score 0.5 probability.
    """
    result = np.empty(0, dtype=int)
    for i in range(len(probas)):
        probability = probas[i]
        if probability[0] > thresh:
            result = np.append(result, 0)
        elif probability[1] > thresh:
            result = np.append(result, 1)
        elif probability[0] != probability[1]:
            result = np.append(result, -1)
        else:
            msg = f"Something is wrong with the probability at index: {i}!"
            raise ValueError(msg)
    return result


def predict_proba(
        classifier: PredictorProto,
        test_x: np.ndarray | list[Verse],
        test_labels: np.ndarray,
        threshold: float = 0.5,
    ) -> ProbaPredictions:
    """Predict on the data with the classifier and return some statistics.

    Args:
        classifier: any object that has a method `.predict_proba()`.
        test_x: inputs to the classifier
        test_labels: correct labels (gold references) for the inputs. This is
            not used to predict probabilities, but is used to calculate
            prediction accuracy etc.
        threshold: the threshold for probability to be counted as a label for
            a particular class.

    Returns:
        a :class:`ProbaPredictions` instance.
    """
    y_pred_proba = classifier.predict_proba(test_x)
    # convert the list of probas to a list of labels
    y_pred = _convert(y_pred_proba, threshold)

    num_mislabels, num_correct = quick_stats(test_x, test_labels, y_pred)

    preds = ProbaPredictions(test_x, y_pred, test_labels,
                     num_correct, num_mislabels)
    preds.set_probas(y_pred_proba)
    return preds


def metricise(
        y_true: list[int] | np.ndarray,
        y_pred_pos: list[int] | np.ndarray | None = None,
        y_pred_neg: list[int] | np.ndarray | None = None,
        y_all: list[int] | np.ndarray | None = None,
        y_probas: list[list[float]] | np.ndarray | None = None
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate performance metrics of a classifier using the outputs.

    This function works when::

        1. ``y_all`` is defined
        OR
        2. ``y_pred_pos`` and ``y_pred_neg`` are defined

    .. note::
        if ``y_all``, ``y_pred_pos``, and ``y_pred_neg`` are set,
        ``y_all`` takes precedence and only their values will be used.

    Args:
        y_true: Correct (gold reference) labels
        y_pred_pos: predictions for the positive class instances (tp + fn)
        y_pred_neg: predictions for the negative class instances (tn + fp)
        y_all: all predictions in a one-dimensional list.
        y_probas: probabilities predicted for each sample.

    Returns:
        accuracy, recall, and f1 scores as floats.

    Raises:
        ValueError: if neither y_all or a pair (y_pred_pos, y_pred_neg) is
            given since there is nothing to calculate scores from
    """
    y_pred = []
    if y_all is not None:
        y_pred = y_all
    elif y_pred_pos is not None and y_pred_neg is not None:
        y_pred.extend(y_pred_pos)
        y_pred.extend(y_pred_neg)
    else:
        msg = (f"Define (y_pred_neg (currently {y_pred_neg}) "
               + f"and y_pred_pos (currently {y_pred_pos})) "
               + f"OR y_all (currently {y_all}).")
        raise ValueError(msg)

    y_pred = np.array(y_pred)

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

    # if y_probas is not None:
    #     y_probas_pos = [proba[1] for proba in y_probas]
    #     cel = log_loss(y_true, y_probas_pos)  # cross-entropy loss
    #     print("log loss > ")
    #     print(cel)
    #     roc_auc = roc_auc_score(y_true, y_probas_pos)
    #     print(f"Roc AUC: {roc_auc}")

    return (accuracy, precision, recall, fbeta)


def plot_charts(
                y_true: list[int] | np.ndarray,
                y_pred: list[int] | np.ndarray,
                y_probas: list[list[float]] | np.ndarray,
            ) -> None:
    """Plot charts given an estimator and its predictions."""
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


def _sort_arrays(
        objs: np.ndarray, vals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    """Sort the two arrays into the same order.

    Args:
        objs: np.ndarray holding any objects.
        vals: np.ndarray with values that can be sorted.

    Returns:
        Two np.ndarrays sorted in the same way.
        They will be ordered in an ascending manner (small -> big).
    """
    sort_ids = np.argsort(vals)
    sorted_objs = objs[sort_ids]
    sorted_cnts = vals[sort_ids]
    return (sorted_objs, sorted_cnts)


def _sort_counter(cnts: Counter) -> tuple[np.ndarray, np.ndarray]:
    """Sort the Counter contents by the counts, return them as np array.

    Args:
        cnts: a :class:`Counter` class object

    Returns:
        Two sorted np.ndarrays for the Counter keys and values.
        They will be ordered in an ascending manner (small -> big).
    """
    cnt_targets = list(cnts.keys())
    cnt_vals = list(cnts.values())
    return _sort_arrays(np.array(cnt_targets), np.array(cnt_vals))


def find_top_k_words(
    verses: list[Verse],
    top_k: int = 150,
    save_file: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Find the top k words from the verses.

    Args:
        verses: list of Verse objects
        top_k: number of top k n_grams to discover
        save_file: Name of the file to save.
            Only works when syr_vocabs is provided.

    Returns:
        A tuple of (top k n_grams in Syriac script,
                    top k transliterated n_grams,
                    counts of the top k n_grams.
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

    sorted_translits, sorted_tl_cnts = _sort_counter(Counter(translit_words))

    # revert the order of the sorted arrays
    top_cnts = sorted_tl_cnts[::-1][:top_k]
    top_translits = sorted_translits[::-1][:top_k]

    # validate Syriac script version is available
    if len(syr_words) > 0:
        sorted_syr_words, sorted_syr_cnts = _sort_counter(Counter(syr_words))
        # If Syriac version is provided but word counts do not match up
        # with the transliterated version, raise an exception
        if sorted_syr_cnts.all() != sorted_tl_cnts.all():
            msg = ("Syriac data for words were provided, but the counts of "
                   + "each word from Syriac and Transliterated data "
                   + "do not match!")
            raise ValueError(msg)
        # revert the order of the sorted array
        top_syr_words = sorted_syr_words[::-1][:top_k]
    else:
        top_syr_words = None

    # print(top_cnts[:10])
    # print(top_syr_words[:10])
    # print(top_translits[:10])

    # Save the words to a file
    if save_file is not None and top_syr_words is not None:
        csv_data = "Syriac,Transliteration,Counts"
        for i in range(top_k):
            csv_data += (f"'{top_syr_words[i]}',"
                        f"{top_translits[i]!s},{int(top_cnts[i])}")
    elif save_file is not None:
        # Format the data without Syriac script
        csv_data = "Transliteration,Counts"
        for i in range(top_k):
            csv_data += (f"{top_translits[i]!s},{int(top_cnts[i])}")

        Path(save_file).write_text(csv_data, encoding="utf-8")
    return (top_syr_words, top_translits, top_cnts)


def get_top_n_grams(
        translit_vocabs: Counter,
        syr_vocabs: Counter | None = None,
        top_k: int = 150,
        save_file: str | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Get top k n-grams, based on the counts stored in translit_vocabs.

    Args:
        translit_vocabs: Counter object for top k vocabs
        syr_vocabs: Couner object for top k Syriac vocabs
        top_k: number of top k n_grams to discover
        save_file: Name of the file to save.
            Only works when syr_vocabs is provided.

    Returns:
        A tuple of (top k n_grams in Syriac script,
                    top k transliterated n_grams,
                    counts of the top k n_grams.
                    )

    Raises:
        ValueError: if cnts of Syriac and transliterated n_grams do not match
    """
    sorted_n_grams, sorted_cnts = _sort_counter(translit_vocabs)

    top_n_grams = sorted_n_grams[::-1][:top_k]
    top_cnts = sorted_cnts[::-1][:top_k]

    # print(top_cnts[:10])
    # print(top_n_grams[:10])

    if syr_vocabs is not None:
        sorted_syr, sorted_syr_cnts = _sort_counter(syr_vocabs)

        if sorted_syr_cnts.all() != sorted_cnts.all():
            msg = (
                "The numbers of counted objects found in"
                + " translit_vocabs and syr_vocabs do not match."
            )
            raise ValueError(msg)

        top_targets_syr = sorted_syr[::-1][:top_k]

        if save_file is not None:
            csv_data = "Syriac,Transliteration,Counts"
            for i in range(top_k):
                csv_data += f"'{''.join(top_targets_syr[i])}',"
                csv_data += f"'{top_n_grams[i]!s}',{int(top_cnts[i])}"
            # print(csv_data.split("")[1])
            Path(save_file).write_text(csv_data, encoding="utf-8")
        return (top_targets_syr, top_n_grams, top_cnts)
    return (None, top_n_grams, top_cnts)


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
    ot_test_x: list[Verse],
    nt_test_x: list[Verse],
    *,
    plot: bool = False,
    ) -> tuple[
            ProbaPredictions, ProbaPredictions,
            Mislabels, Mislabels]:
    """Evaluate a classifier with provided test sets.

    Args:
        clf: a BoWEstimator instance.
        ot_test_x: list of OT verses to evaluate on.
        nt_test_x: list of NT verses to evaluate on.
        plot: if True, create charts
            and display them with :func:`classifier.eval_utils.plot_charts`.

    Returns:
        :class:`ProbaPredictions` instances, one for OT and another for NT,
        along with :class:`Mislabels` instances for OT and NT.
    """
    # Generate gold reference labels
    ot_test_y = [0 for x in ot_test_x]
    nt_test_y = [1 for x in nt_test_x]
    # concatenate gold references for metricise function
    all_test_y = []
    all_test_y.extend(ot_test_y)
    all_test_y.extend(nt_test_y)
    # cast / convert to np.ndarray
    all_test_y = np.array(all_test_y)
    print(f"test size: {all_test_y.shape}")

    # predict probabilities with clf
    print("OT --->")
    ot_proba_preds = predict_proba(
            clf,  # type: ignore[reportArgumentType]
        ot_test_x, np.array(ot_test_y)
    )
    print("NT --->")
    nt_proba_preds = predict_proba(
        clf,  # type: ignore[reportArgumentType]
        nt_test_x, np.array(nt_test_y)
    )
    print("All --->")
    y_all = ot_proba_preds.predictions
    y_all = np.append(y_all, nt_proba_preds.predictions)
    probas = ot_proba_preds.get_probas()
    probas = np.append(probas, nt_proba_preds.get_probas())
    metricise(all_test_y, y_all=y_all, y_probas=probas)

    if plot:
        plot_charts(all_test_y, y_all, probas)

    # figure out which verses the classifier mislabelled
    # reportArgumentType can be disabled here because ProbaPredictions.samples
    # will always be ``list[Verse]`` since we give predict_proba function
    # ``{nt/ot}_test_x``, which are both ``list[Verse]`` type.
    ot_mislabels = find_mislabels(
        ot_test_y, ot_proba_preds.predictions,
        test_x=ot_proba_preds.samples, probas=ot_proba_preds.get_probas()  # type: ignore[reportArgumentType]
        )
    nt_mislabels = find_mislabels(
        nt_test_y, nt_proba_preds.predictions,
        test_x=nt_proba_preds.samples, probas=nt_proba_preds.get_probas()  # type: ignore[reportArgumentType]
        )
    return (ot_proba_preds, nt_proba_preds, ot_mislabels, nt_mislabels)


def eval_and_save(  # noqa: PLR0913
    clf: BoWEstimator,
    ot_test_samples: list,
    nt_test_samples: list,
    file_formatter: Callable,
    *,
    out_dir: str = "./out/",
    save_file_prefix: str = "",
    save_file_suffix: str = "",
    save_file_ext: str = ".csv",
) -> tuple[BoWEstimator, list[ProbaPredictions]]:
    """Wrapper around evaluate_classifier, save_mislabels, and save_all_preds.

    Args:
        clf: a classifier wrapped in :class:`BoWEstimator`
        ot_test_samples: inputs to the classifier from OT
        nt_test_samples: inputs to the classifier from NT
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

    Returns:
        A BoWEstimator instance and two ProbaPredictions, each for OT and NT.

    .. note::
        This function uses :meth:`pathlib.Path.write_text` directly.
        Error descriptions in this documentation are not thorough.

    .. seealso::
        :func:`classifier.eval_utils.evaluate_classifier`
            for params: ``clf``, ``ot_test_X``, ``ot_test_y``,
            ``nt_test_X``, ``nt_test_y``
        :func:`classifier.eval_utils.save_mislabels`
        :func:`classifier.eval_utils.save_all_preds`
            for param: ``formatter``.
    """
    # convert the out_dir to Path
    out_dir_p = Path(out_dir)
    # Remove prepended slash in save_file_prefix
    # since they break Path concatenation
    if save_file_prefix[0] == "/":
        save_file_prefix = save_file_prefix[1:]

    # evaluate the classifier with the provided samples
    (ot_probas, nt_probas, ot_mislabels, nt_mislabels) = evaluate_classifier(
        clf, ot_test_samples, nt_test_samples
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
