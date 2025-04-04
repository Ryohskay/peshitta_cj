from sklearn.metrics import accuracy_score, auc, ConfusionMatrixDisplay, f1_score, recall_score
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from classifier.fitting_utils import BoW_Estimator

import matplotlib.pyplot as plt
import numpy as np
from typing import Union


def quick_stats(
        inputs: np.ndarray,
        y_correct: np.ndarray,
        prediction: np.ndarray,
    ) -> tuple[int, int]:
    """Calculate and print some basic statistics on model inputs & outputs."""
    sample_size = 0
    try:
        sample_size = inputs.shape[0]
    except AttributeError:
        sample_size = len(inputs)

    if sample_size == 0:
        msg = "Cannot measure the size of model input array X!"
        raise ValueError(msg)

    num_mislabels = ((y_correct != prediction).sum())
    num_correct = sample_size - num_mislabels

    print(f"Number of mislabeled points out of the total {sample_size} verses: "
          + f"{num_mislabels}")
    print(f"Local accuracy: {accuracy_score(y_correct, prediction):.02f}")

    return (num_mislabels, num_correct)


def find_mislabels(
        y_correct: list,
        y_pred: list,
        X: None | list=None,
        probas: list=None,
    ) -> (list, list, None | list, None | list):
    """Find and return mislabelled verses in string format."""
    idcs = list()
    mislabels = list()
    correct_labels = list()
    mislabel_verses = list()
    mislabel_probas = list()

    for i in range(len(y_correct)):
        if y_correct[i] != y_pred[i]:
            idcs.append(i)
            mislabels.append(y_pred[i])
            correct_labels.append(y_correct[i])

    if X is not None and isinstance(X, np.ndarray):
        # give the contents of mislabelled verses
        mislabel_verses = X[idcs]
    elif X is not None:
        # give the contents of mislabelled verses
        # in case X is not np array
        verses = []
        for idx in idcs:
            verses.append(X[idx])
        mislabel_verses = verses

    if probas is not None:
        mislabel_probas = np.array(probas)[idcs]

    return (idcs, mislabels, correct_labels, mislabel_verses, mislabel_probas)


def predict(
        classifier: type,
        test_X: np.ndarray | list,
        test_labels: np.ndarray
    ) -> (np.ndarray, np.ndarray, np.ndarray):
    """Predict on the data with a given classifier, and return some simple statistics.

    The classifier must have a method `.predict()`.
    """
    y_pred = classifier.predict(test_X)
    num_mislabels, num_correct = quick_stats(test_X, test_labels, y_pred)
    
    return (y_pred, num_mislabels, num_correct)


def _convert(probas: np.array) -> np.array:
    result = np.empty(0, dtype=int)
    """Convert list of probabilities to a list of labels."""
    for i in range(len(probas)):
        probability = probas[i]
        if probability[0] > 0.5:
            result = np.append(result, 0)
        elif probability[1] > 0.5:
            result = np.append(result, 1)
        elif probability[0] == probability[1]:
            result = np.append(result, None)
        else:
            raise ValueError(f"Something is wrong with the probability at index: {i}!")
    return result


def predict_proba(
        classifier: type,
        test_X: np.ndarray | list,
        test_labels: np.array
    ) -> (np.array, np.array, np.array, np.array):
    """Predict on the data with the classifier, and return some simple statistics.

    The classifier must have a method `.predict_proba()`.
    """
    y_pred_proba = classifier.predict_proba(test_X)
    y_pred = _convert(y_pred_proba) # convert the list of probas to a label

    num_mislabels, num_correct = quick_stats(test_X, test_labels, y_pred)
    
    return (y_pred_proba, y_pred, num_mislabels, num_correct)


def metricise(
        y_true: list[int],
        y_pred_pos: list[int] | None=None,
        y_pred_neg: list[int] | None=None,
        y_all: list[int] | None=None,
        conf_m: bool=False,  
    ) -> tuple[float, float, float]:
    """Calculate performance metrics of a classifier using the outputs.

    conf_m:
        if True, then display confusion matrix.
    """
    y_pred = []
    if y_all is None and y_pred_pos is not None and y_pred_neg is not None:
        y_pred.extend(y_pred_pos)
        y_pred.extend(y_pred_neg)
    elif y_all is not None:
        y_pred = y_all
    else:
        msg = "Define (y_pred_neg and y_pred_pos) OR y_all."
        raise ValueError(msg)

    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1_result = f1_score(y_true, y_pred)

    if conf_m:
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        plt.show()
    
    print(f"Overall Accuracy: {accuracy:.02f}")
    print(f"Overall Recall: {recall:.02f}")
    print(f"F1 Score: {f1_result:.02f}")
    return (accuracy, recall, f1_result)


def _sort_arrays(objs: np.array, vals: np.array):
    """Sort the two arrays into the same order."""
    sort_ids = np.argsort(vals)
    sorted_objs = objs[sort_ids]
    sorted_cnts = vals[sort_ids]
    return (sorted_objs, sorted_cnts)


def _sort_counter(cnts: Counter):
    """Sort the Counter contents by the counts, return them as np array."""
    cnt_targets = list(cnts.keys())
    cnt_vals = list(cnts.values())
    return _sort_arrays(np.array(cnt_targets), np.array(cnt_vals))


def find_top_k_words(
        verses: list[tuple[str, list[str], list[str]]],
        top_k: int=150,
        save_file: None | str=None,
    ) -> (np.array, np.array, np.array):
    """Find the top k words from the verses."""
    syr_words = []
    translit_words = []

    for vrs in verses:
        syr_words.extend(vrs[2])
        translit_words.extend(vrs[1])

    sorted_syr_words, sorted_syr_cnts = _sort_counter(Counter(syr_words))
    sorted_translits, sorted_tl_cnts = _sort_counter(Counter(translit_words))

    if sorted_syr_cnts.all() != sorted_tl_cnts.all():
        msg = "Word counts for Syriac and Transliterated data do not match!"
        raise ValueError(msg)

    top_cnts = sorted_syr_cnts[::-1][:top_k]
    top_syr_words = sorted_syr_words[::-1][:top_k]
    top_translits = sorted_translits[::-1][:top_k]
    
    print(top_cnts[:10])
    print(top_syr_words[:10])
    print(top_translits[:10])

    if save_file is not None:
        csv_data = "Syriac,Transliteration,Counts\n"
        for i in range(top_k):
            csv_data += f"'{str(top_syr_words[i])}',"
            csv_data += f"{str(top_translits[i])},{int(top_cnts[i])}\n"

        Path(save_file).write_text(csv_data)
    return (top_syr_words, top_translits, top_cnts)


def get_top_n_grams(
        translit_vocabs: Counter,
        syr_vocabs: None | Counter=None,
        top_k: int=150,
        save_file: None | str=None,
    ) -> (Union[np.array,None], np.array, np.array):
    """Get top k n-grams, based on the counts stored in translit_vocabs.

    save_file: str
        Only works when syr_vocabs is provided.
    """
    sorted_n_grams, sorted_cnts = _sort_counter(translit_vocabs)

    top_n_grams = sorted_n_grams[::-1][:top_k]
    top_cnts = sorted_cnts[::-1][:top_k]

    print(top_cnts[:10])
    print(top_n_grams[:10])

    if syr_vocabs is not None:
        sorted_syr, sorted_syr_cnts = _sort_counter(syr_vocabs)

        if sorted_syr_cnts.all() != sorted_cnts.all():
            msg = ("The numbers of counted objects found in"
                   + " translit_vocabs and syr_vocabs do not match.")
            raise ValueError(msg)

        top_targets_syr = sorted_syr[::-1][:top_k]

        if save_file is not None:
            csv_data = "Syriac,Transliteration,Counts\n"
            for i in range(top_k):
                csv_data += f"'{''.join(top_targets_syr[i])}',"
                csv_data += f"'{str(top_n_grams[i])}',{int(top_cnts[i])}\n"
            # print(csv_data.split("\n")[1])
            Path(save_file).write_text(csv_data)
        return (top_targets_syr, top_n_grams, top_cnts)
    return (None, top_n_grams, top_cnts)


def split_list(lis: list, parts: int=5) -> list[list]:
    """Split a list into sub-arrays of even size."""
    if len(lis) % parts != 0:
        print(f"NOTE: the number of training samples ({len(lis)}) is not divisible by {parts}.")
        print("Resulting split of sub-arrays will be uneven.")

    if len(lis) < parts:
        msg = f"Cannot devide a list of length {len(lis)} into {parts} parts!"
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
    split_end = split_ids[0]-1
    for i in split_ids:
        results.append(lis[split_start:int(split_end)])
        split_start = i-1
        split_end += separator
    results.append(lis[split_start:])
    return results



def evaluate_classifier(
        clf: BoW_Estimator,
        ot_test_X: list,
        ot_test_y: list,
        nt_test_X: list,
        nt_test_y: list,
        show_confusion_matrix: bool=False,
    ) -> (list[(float, float)], list[(float, float)]):
    """Evaluate a classifier with provided test sets."""
    print("OT --->")
    ot_probas, ot_y_preds, _, _  = predict_proba(
                                                 clf,
                                                 ot_test_X,
                                                 np.array(ot_test_y)
                                                 )
    print("NT --->")
    nt_probas, nt_y_preds, _, _ = predict_proba(
                                                clf,
                                                nt_test_X,
                                                np.array(nt_test_y)
                                                )
    print("All --->")
    all_test_y = []
    all_test_y.extend(ot_test_y)
    all_test_y.extend(nt_test_y)
    all_test_y = np.array(all_test_y)
    metricise(all_test_y, ot_y_preds, nt_y_preds, conf_m=show_confusion_matrix)

    # figure out which verses the classifier mislabelled
    ot_mislabels = find_mislabels(ot_test_y, ot_y_preds, X=ot_test_X, probas=ot_probas)
    nt_mislabels = find_mislabels(nt_test_y, nt_y_preds, X=nt_test_X, probas=nt_probas)
    return (ot_probas, nt_probas, ot_mislabels, nt_mislabels)


def save_mislabels(
        ot_mislabels: tuple,
        nt_mislabels: tuple,
        formatter: Callable,
        ot_save_file: str | Path="./out/prediction_mislabels_ot.csv",
        nt_save_file: str | Path="./out/prediction_mislabels_nt.csv",
    ):
    """Save mislabelled verses into a file.

    formatter: Callable
        any callable object (function, method, etc.)
        that returns a formatted string which can be directly
        written to a file.
    """
    ot_data = formatter(
                X=ot_mislabels[3],
                probas=ot_mislabels[4],
                y_correct=ot_mislabels[2],
            )

    nt_data = formatter(
                X=nt_mislabels[3],
                probas=nt_mislabels[4],
                y_correct=nt_mislabels[2],
            )

    # write formatted texts to files
    Path(ot_save_file).write_text(ot_data)
    Path(nt_save_file).write_text(nt_data)


def save_all_preds(
        ot_test_X: list,
        ot_probas: list,
        ot_test_y: list,
        nt_test_X: list,
        nt_probas: list,
        nt_test_y: list,
        formatter: Callable,
        ot_save_file: str | Path="./out/prediction_all_ot.csv",
        nt_save_file: str | Path="./out/prediction_all_nt.csv",
    ):
    """Save all verses into a file, along with prediction results.

        :param Callable formatter: any callable object (function, method, etc.)
            that returns a formatted string which can be directly
            written to a file.

        :note: This function uses pathlib.Path().write_text directly.
            Error descriptions in this documentation are not thorough.
        
        :raises FileNotFoundError: [Errno 2] No such file or directory

        :raises OSError: [Errno 30] Read only file system
            (raised if no permission to write)
    """
    # save all prediction results to csv files
    ot_csv = formatter(X=ot_test_X, probas=ot_probas, y_correct=ot_test_y)
    nt_csv = formatter(X=nt_test_X, probas=nt_probas, y_correct=nt_test_y)

    # write CSV-formatted texts to files
    Path(ot_save_file).write_text(ot_csv)
    Path(nt_save_file).write_text(nt_csv)


def eval_and_save(
        clf: BoW_Estimator,
        ot_test_X: list,
        ot_test_y: list,
        nt_test_X: list,
        nt_test_y: list,
        formatter: Callable,
        out_dir: str | Path=(PROJ_ROOT/"out/"),
        save_file_prefix: str="",
        save_file_suffix: str="",
        save_file_ext: str=".csv"
        ):
    """Wrapper around evaluate_classifier, save_mislabels, and save_all_preds.

    See evaluate_classifier for params: clf, ot_test_X, ot_test_y,
                                        nt_test_X, nt_test_y
    See save_mislabels & save_all_preds for param: formatter

    out_dir: str | Path
        Path or string of path to the directory to save result files.

    save_file_prefix, save_file_suffix: str
        Prefixes and suffixes to add before/after the default file name for each
        evaluation process. These are used to construct save file names
        passed to save_mislabels and save_all_preds functions.

    ERRORS:
        This function uses pathlib.Path().write_text directly.
        Thus, it may return e.g.:
            FileNotFoundError
                [Errno 2]
                No such file or directory

            OSError
                [Errno 30]
                Read only file system (Occurs if no permission to write)
    """
    (ot_probas, nt_probas,
     ot_mislabels, nt_mislabels) = evaluate_classifier(
                                                clf,
                                                ot_test_X,
                                                ot_test_y,
                                                nt_test_X,
                                                nt_test_y
                                           )

    # Construct save files' paths
    ot_mislabels_file = (save_file_prefix + "prediction_mislabels_ot"
                        + save_file_suffix + save_file_ext)
    nt_mislabels_file = (save_file_prefix + "prediction_mislabels_nt"
                        + save_file_suffix + save_file_ext)
    ot_all_file = (save_file_prefix + "prediction_all_ot"
                        + save_file_suffix + save_file_ext)
    nt_all_file = (save_file_prefix + "prediction_all_nt"
                        + save_file_suffix + save_file_ext)

    # Save the evaluation results to files
    save_mislabels(ot_mislabels, nt_mislabels,
                   formatter=formatter,
                   ot_save_file=(out_dir/ot_mislabels_file),
                   nt_save_file=(out_dir/nt_mislabels_file),
                   )

    save_all_preds(ot_test_X, ot_probas, ot_test_y,
                   nt_test_X, nt_probas, nt_test_y,
                   formatter=formatter,
                   ot_save_file=(out_dir/ot_all_file),
                   nt_save_file=(out_dir/nt_all_file)
                   )
