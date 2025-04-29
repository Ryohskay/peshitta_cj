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

from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator

from src.classifier.cal_aa_eval import remove_proper_nouns as nopropn_cal
from src.classifier.cal_aa_eval import remove_underscores, run_cal_clf
from src.classifier.dataset_skeleton import LoadedDataset
from src.classifier.etcbc_aa_eval import remove_non_chars, run_etcbc_clf
from src.classifier.etcbc_aa_eval import remove_proper_nouns as nopropn_etcbc
from src.classifier.eval_utils import (
    cross_validate,
)
from src.classifier.fitting_utils import identity
from src.classifier.fname_utils import FnameExtraOpts, SavefileName
from src.classifier.textfabric_utils import load_etcbc_dataset
from src.classifier.wrappers import BoWEstimator
from src.shared.classification_algos import get_algo_by_name


def cross_validate_clf(
    algo: BaseEstimator,
    n_window_max: int,
    ds: LoadedDataset,
    formatter: Callable[[list[str]], str] = " ".join,
    preprocessor: Callable[[list[str]], list[str]] | None = None,
    **kwargs,
) -> tuple[int, float, list[dict]]:
    """Cross-validate the classifier with different n-gram spans.

    Returns:
        index of the best performing classifier, the best f-score, and a list of
        configurations for each classifier.
    """
    avg_f_scores = []
    configs = []
    for n_window in range(1, n_window_max + 1):
        classifier = BoWEstimator(algo, formatter, n=n_window)
        if preprocessor is not None:
            classifier.set_preprocessor(preprocessor)
            # print(f"preprocessor: {preprocessor.__name__}")  # debug
        _, _, _, fb = cross_validate(
            classifier,
            ds.train.get_samples(),
            ds.train.get_labels(),
        )
        # print(
        #     f"n_window={n_window}, fb={fb}, shape={np.array(fb).shape}"
        # )  # debug
        # if len(np.array(fb, dtype=np.float64).shape) == 3:
        #     avg_f_scores.append(
        #         np.average([score for v in fb for vals in v
        #                           for score in vals])
        #     )
        if len(np.array(fb, dtype=np.float64).shape) == 2:
            avg_f_scores.append(np.average([score for v in fb for score in v]))
        else:
            msg = (
                "Unexpected shape of the cross-validation results: "
                + f"{np.array(fb).shape}"
            )
            raise ValueError(msg)
        configs.append({"n_window": n_window, "other_opts": kwargs})
    max_idx = np.argmax(avg_f_scores)
    return int(max_idx), avg_f_scores[max_idx], configs


def find_best_clf(
    clf_alias: str,
    n_window_max: int,
    ds: LoadedDataset,
    base_fname: SavefileName,
    **kwargs,
) -> tuple[float, float]:
    """Find the best performing classifier for a given dataset and run it.

    Args:
        clf_alias (str): The alias of the classifier to use.
        n_window_max (int): The maximum n-gram span to consider.
        ds (LoadedDataset): The dataset to use for training and testing.
        base_fname (SavefileName): The base filename for saving results.
        **kwargs: Additional arguments for the classifier.

    Returns:
        tuple[float, float]: The best f-scores for character and word n-grams.
    """
    algo = get_algo_by_name(clf_alias, **kwargs)
    eval_clf = run_cal_clf if base_fname.origin == "CAL" else run_etcbc_clf
    preprocessor = (
        remove_underscores if base_fname.origin == "CAL" else remove_non_chars
    )
    data_map_func = nopropn_cal if base_fname.origin == "CAL" else nopropn_etcbc
    # Remove personal names and place names from
    # both the training and test datasets
    ds.train.map_on_samples(data_map_func)
    ds.test.map_on_samples(data_map_func)
    # define the save directory for the results
    local_save_dir = (
        Path("out/cross_validation")
        if "local_save_dir" not in kwargs
        else kwargs["local_save_dir"]
    )

    # find the best performing classifier for the character n-grams
    max_idx, max_c_fscore, configs = cross_validate_clf(
        algo, n_window_max, ds, preprocessor=preprocessor
    )

    # Save configs for the best performing classifier
    msg = (
        "Best performing char-ngram classifier: "
        + f"{configs[max_idx]} with f-score {max_c_fscore}"
    )
    base_fname.ext = "txt"
    base_fname.set_ngram_opts(
        n=configs[max_idx]["n_window"], is_char_level=True
    )
    local_save = local_save_dir / ("best_clf_" + base_fname.get_fname())
    with local_save.open("a+", encoding="utf-8") as f:
        f.write(msg + "\n")
        print(msg)
    # exit()

    n_window = configs[max_idx]["n_window"]

    # apply the best performing classifier to the test set
    # and save the results
    print(f"Char n-gram classifier with n_window={n_window}")
    print("\n>> Remove PN, GN, underscores from both training & test sets")
    # and train new classifiers
    c_clf_rnus = BoWEstimator(algo, " ".join, n=n_window)
    c_clf_rnus.set_preprocessor(preprocessor)

    save_fname_rnus = base_fname.copy()
    save_fname_rnus.set_ngram_opts(n=n_window, is_char_level=True)
    save_fname_rnus.add_extra_opts(
        [
            FnameExtraOpts.REMOVE_UNDERSCORES,
            FnameExtraOpts.REMOVE_PROPN,
            FnameExtraOpts.REMOVE_FROM_BOTH,
        ]
    )

    eval_clf(
        clf_alias,
        n_window,
        ds,
        base_fname,
    )

    # find the best performing classifier for the word n-grams
    max_idx, max_w_fscore, configs = cross_validate_clf(
        algo, n_window_max, ds, identity, preprocessor
    )
    n_window = configs[max_idx]["n_window"]
    # Save configs for the best performing classifier
    msg = (
        "Best performing word-ngram classifier: "
        + f"{configs[max_idx]} with f-score {max_c_fscore}"
    )
    base_fname.ext = "txt"
    base_fname.set_ngram_opts(
        n=configs[max_idx]["n_window"], is_char_level=False
    )
    local_save = local_save_dir / ("best_clf_" + base_fname.get_fname())
    with local_save.open("a+", encoding="utf-8") as f:
        f.write(msg + "\n")
        print(msg)
    print(f"\n=============== Word {n_window}-grams =================")
    print("\n>> Remove PN, GN from both training & test sets")
    # Remove personal names and place names from
    # both the training and test datasets
    # and train new classifiers
    w_clf_rnus = BoWEstimator(algo, identity, n=n_window)
    w_clf_rnus.set_preprocessor(preprocessor)

    save_fname_rnus_w = base_fname.copy()
    save_fname_rnus_w.set_ngram_opts(n=n_window, is_char_level=False)
    save_fname_rnus_w.add_extra_opts(
        [
            FnameExtraOpts.REMOVE_UNDERSCORES,
            FnameExtraOpts.REMOVE_PROPN,
            FnameExtraOpts.REMOVE_FROM_BOTH,
        ]
    )

    eval_clf(
        clf_alias,
        n_window,
        ds,
        save_fname_rnus_w,
    )
    return max_c_fscore, max_w_fscore


if __name__ == "__main__":
    etcbc_ds = load_etcbc_dataset()
    n_max = 6

    print("> ETCBC")
    print("\n================== MNB RESULTS================")
    clf_alias = "mnb"
    base_fname_mnb = SavefileName("ETCBC", clf_alias)
    find_best_clf(clf_alias, n_max, etcbc_ds, base_fname_mnb)

    print("\n============== K-NEAREST NEIGHBOURS: UNIFORM ==================\n")
    base_fname_knn = SavefileName("ETCBC", "knn")
    max_char_scores = []
    max_word_scores = []
    for k in range(3, 10):
        base_fname_knn.set_knn_opts(k=k)
        print(f"\nKNN Classifier with k={k}")
        max_char_score, max_w_score = find_best_clf(
            "knn", n_max, etcbc_ds, base_fname_knn, n_neighbors=k
        )
        max_char_scores.append(max_char_score)
        max_word_scores.append(max_w_score)
    argmax_char = np.argmax(max_char_scores)
    argmax_word = np.argmax(max_word_scores)
    print(
        f"\nBest char n-gram KNN classifier: {max_char_scores[argmax_char]} "
        + f"with k={argmax_char + 1}"
    )
    print(
        f"\nBest word n-gram KNN classifier: {max_word_scores[argmax_word]} "
        + f"with k={argmax_word + 1}"
    )

    print("\n============== K-NEAREST NEIGHBOURS: DISTANCE =================\n")
    base_fname_knn = SavefileName("ETCBC", "knn")
    max_char_scores = []
    max_word_scores = []
    for k in range(3, 10):
        base_fname_knn.set_knn_opts(k=k, weights="distance")
        print(f"\nKNN Classifier with k={k}")
        max_char_score, max_w_score = find_best_clf(
            "knn",
            n_max,
            etcbc_ds,
            base_fname_knn,
            n_neighbors=k,
            weights="distance",
        )
        max_char_scores.append(max_char_score)
        max_word_scores.append(max_w_score)
    argmax_char = np.argmax(max_char_scores)
    argmax_word = np.argmax(max_word_scores)
    print(
        f"\nBest char n-gram KNN classifier: {max_char_scores[argmax_char]} "
        + f"with k={argmax_char + 1}"
    )
    print(
        f"\nBest word n-gram KNN classifier: {max_word_scores[argmax_word]} "
        + f"with k={argmax_word + 1}"
    )

    print("\n=========================== RANDOM FOREST =====================\n")
    base_fname_rf = SavefileName("ETCBC", "rf")
    max_char_scores = []
    max_word_scores = []
    for n_tree in range(100, 501, 100):
        base_fname_rf.set_rf_opts(n_estimators=n_tree)
        print(f"\nRandom Forest Classifier with {n_tree} trees")
        max_char_score, max_w_score = find_best_clf(
            "rf", n_max, etcbc_ds, base_fname_rf, n_estimators=n_tree
        )
        max_char_scores.append(max_char_score)
        max_word_scores.append(max_w_score)
    argmax_char = np.argmax(max_char_scores)
    argmax_word = np.argmax(max_word_scores)
    print(
        "\nBest char n-gram Random Forest classifier: "
        + f"{max_char_scores[argmax_char]} with n_estimators="
        + f"{list(range(100, 501, 100))[argmax_char]}"
    )
    print(
        "\nBest word n-gram Random Forest classifier: "
        + f"{max_word_scores[argmax_word]} with "
        + f"n_estimators={list(range(100, 501, 100))[argmax_char]}"
    )

    print("\n===================SVC====================\n")
    base_fname_svc = SavefileName("ETCBC", "svc")
    max_char_scores = []
    max_word_scores = []
    for c in [0.1, 1, 10, 100]:
        base_fname_svc.set_svc_opts(c=c)
        print(f"\nSVC Classifier with C={c}")
        max_char_score, max_w_score = find_best_clf(
            "svc", n_max, etcbc_ds, base_fname_svc, c=c
        )
        max_char_scores.append(max_char_score)
        max_word_scores.append(max_w_score)
    argmax_char = np.argmax(max_char_scores)
    argmax_word = np.argmax(max_word_scores)
    print(
        f"\nBest char n-gram SVC classifier: {max_char_scores[argmax_char]} "
        + f"with C={[0.1, 1, 10, 100][argmax_char]}"
    )
    print(
        f"\nBest word n-gram SVC classifier: {max_word_scores[argmax_word]} "
        + f"with C={[0.1, 1, 10, 100][argmax_word]}"
    )

    print("\n=================MLP====================\n")
    base_fname_mlp = SavefileName("ETCBC", "mlp")
    percepts = [10, 50, 100, 200, 500]
    max_all_char_scores = []
    max_all_word_scores = []
    best_char_percepts_idx = []
    best_word_percepts_idx = []
    for n_layers in [1, 2, 3]:
        max_char_scores = []
        max_word_scores = []
        for h in percepts:
            base_fname_mlp.set_mlp_opts(hidden_layer_sizes=(h,))
            print(f"\nMLP Classifier with {h} hidden units")
            max_char_score, max_w_score = find_best_clf(
                "mlp",
                n_max,
                etcbc_ds,
                base_fname_mlp,
                hidden_layer_sizes=[h] * n_layers,
            )
            max_char_scores.append(max_char_score)
            max_word_scores.append(max_w_score)
        best_char_percepts_idx.append(np.argmax(max_char_scores))
        best_word_percepts_idx.append(np.argmax(max_word_scores))
        max_all_char_scores.append(np.max(max_char_scores))
        max_all_word_scores.append(np.max(max_word_scores))
    argmax_char = np.argmax(max_all_char_scores)
    argmax_word = np.argmax(max_all_word_scores)
    print(
        f"\nBest char n-gram MLP classifier: {max_all_char_scores[argmax_char]}"
        + f" with {percepts[best_char_percepts_idx[argmax_char]]} hidden units"
        + f" and {best_char_percepts_idx[argmax_char] + 1} layers"
    )
    print(
        f"\nBest word n-gram MLP classifier: {max_all_word_scores[argmax_word]}"
        + f" with {percepts[best_word_percepts_idx[argmax_word]]} hidden units "
        + f"and {best_word_percepts_idx[argmax_word] + 1} layers"
    )
