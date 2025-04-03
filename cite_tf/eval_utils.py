from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, recall_score
from collections import Counter
from pathlib import Path

import numpy as np
from typing import Union


def find_mislabels(y_correct: list, y_pred: list, X: None | list=None):
    result = {
            "mislabel_ids": [],
            "mislabels": [],
            "mislabelled_verses": None
            }
    for i in range(len(y_correct)):
        if y_correct[i] != y_pred[i]:
            result["mislabel_ids"].append(i)
            result["mislabels"].append((y_correct[i], y_pred[i]))
    if X is not None:
        # give the contents of mislabelled verses
        result[2] = np.array(X)[result["mislabel_ids"]]
    return result


def predict(
        classifier, test_X: np.ndarray, test_labels: np.array
    ) -> (np.array, np.array, np.array):
    """Predict on the data with a given classifier, and return some simple statistics.

    The classifier must have a method `.predict()`.
    """
    y_pred = classifier.predict(test_X)
    
    num_samples = test_X.shape[0]
    num_mislabels = ((test_labels != y_pred).sum())
    num_correct = num_samples - num_mislabels
    
    print("Number of mislabeled points out of the total %d verses: %d" % (num_samples, num_mislabels))
    print(f"Local accuracy: {accuracy_score(test_labels, y_pred):.02f}")

    return y_pred, num_mislabels, num_correct


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
        classifier: type, test_X: np.ndarray, test_labels: np.array
    ) -> (np.array, np.array, np.array, np.array):
    """Predict on the data with the classifier, and return some simple statistics.

    The classifier must have a method `.predict_proba()`.
    """
    y_pred_proba = classifier.predict_proba(test_X)
    y_pred = _convert(y_pred_proba) # convert the list of probas to a label
    
    num_samples = test_X.shape[0]
    num_mislabels = ((test_labels != y_pred).sum())
    num_correct = num_samples - num_mislabels
    
    print(f"Number of mislabeled points out of the total {num_samples} verses:"
          + f" {num_mislabels}")
    print(f"Local accuracy: {accuracy_score(test_labels, y_pred):.02f}")
    return y_pred_proba, y_pred, num_mislabels, num_correct


def metricise(
        y_true: list[int],
        y_pred_pos: list[int] | None=None,
        y_pred_neg: list[int] | None=None,
        y_all: list[int] | None=None
    ) -> tuple[float, float, float]:
    """Measure the performance of a classifier using the outputs."""
    y_pred = []
    if y_all is None and y_pred_pos is not None and y_pred_neg is not None:
        y_pred.extend(y_pred_pos)
        y_pred.extend(y_pred_neg)
    else:
        y_pred = y_all
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1_result = f1_score(y_true, y_pred)
    
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
