import logging
import matplotlib.pyplot as plt

from src.classifier.result_utils import ResultStatsDict
from src.ui_web.load_clf_stats import JsonifiedSummaryDict, load_clf_stats
from src.ui_web.select_load_classifier import ClassifierConfig, ClassifierResultsModel, ResultFilesIndex, parse_fname


logger = logging.getLogger(__name__)

def plot_f1_scores(
    classifiers_f1_scores: dict[str, list[tuple[int, float]]],
    title: str="F1 Scores"
) -> None:
    """Plot F1 scores of multiple Naïve Bayes classifiers.

    Args:
        classifiers_f1_scores: A dictionary where keys are classifier
                        names and values are lists of tuples (n, f1_score).
        title: Title of the figure.

    Example:
        {
            "Classifier 1": [(1, 0.8), (2, 0.85), (3, 0.9)],
            "Classifier 2": [(1, 0.75), (2, 0.8), (3, 0.85)],
        }
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))

    n_values = None
    for classifier_name, scores in classifiers_f1_scores.items():
        n_values = [n for n, _ in scores]
        f1_scores = [f1 for _, f1 in scores]
        plt.scatter(n_values, f1_scores, marker="o", label=classifier_name)

    if n_values is None or len(n_values) < 1:
        msg = "No values for n of n-grams provided in the input data."
        raise ValueError(msg)

    plt.title(title)
    plt.xlabel("n (n-grams)")
    plt.ylabel("F1 Score")
    plt.xticks(n_values)  # Ensure x-axis ticks match the n values
    plt.ylim(0, 1)  # F1 scores range from 0 to 1
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def avg_f1(
    result_dict: ResultStatsDict,
    target_threshold: float = 0.5,
) -> float:
    """Calculate the average F1 score from the result statistics.

    Args:
        result_dict: Dictionary containing the result statistics.
        target_threshold: The threshold for which to calculate the average F1 score.

    Returns:
        The average F1 score.
    """
    if result_dict is None:
        msg = "The result statistics was not provided."
        raise ValueError(msg)
    supports = result_dict["supports"]
    avg_from = []

    for thresh_stat in result_dict["thresh_stats"]:
        if len(thresh_stat["f_beta"]) != len(supports):
            msg = f"The length of f_beta at threshold {thresh_stat['threshold']} ({len(thresh_stat['f_beta'])}) does not match the number of classes ({len(supports)})."
            logger.warning(msg)
            thresh_stat["f_beta"] = thresh_stat["f_beta"][1:]
        if thresh_stat["threshold"] == target_threshold:
            for i in range(len(thresh_stat["f_beta"])):
                if supports[i] > 0:
                    avg_from.extend([thresh_stat["f_beta"][i]] * supports[i])
    return sum(avg_from) / len(avg_from)

# Example usage
if __name__ == "__main__":
    load_dir = "src/classifier/out/"
    index = ResultFilesIndex(load_dir)
    models: list[ClassifierResultsModel] = []
    for file in index.files:
        if file.is_clf_summary:
            clf_configs = ClassifierConfig(
                name=file.classifier,
                origin=file.origin,
                n=file.n,
                is_n_gram=file.is_n_gram,
                is_bow=file.is_bow,
                is_char_level=file.is_char_level,
                k=file.knn_k,
                weights=file.knn_weights,
                p=file.knn_minkowski_p,
                extra_opts=[opt for opt in file.extra_opts if file.extra_opts[opt]],
            )
            # print(clf_configs)
            models.append(ClassifierResultsModel(clf_configs, index, load_dir))

    clf_f1_scores: dict[str, list[tuple[int, float]]] = {}
    for m in models:
        m.load_results()
        clf_name = m.config.name
        if m.results_summary is not None:
            f1_score = avg_f1(m.results_summary["metrics"])
            if clf_name in clf_f1_scores:
                clf_f1_scores[clf_name].append((m.config.n, f1_score))
            else:
                clf_f1_scores[clf_name] = [(m.config.n, f1_score)]

    plot_f1_scores(clf_f1_scores)
