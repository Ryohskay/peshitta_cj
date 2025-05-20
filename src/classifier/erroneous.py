import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB

from src.classifier.dataset_skeleton import LoadedDataset
from src.classifier.textfabric_utils import load_etcbc_dataset
from src.classifier.wrappers import BoWEstimator  # Assuming this is implemented


def train_multinomial_nb_with_ngrams(
    data: LoadedDataset, n_values: range
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Train MultinomialNB classifiers with n-grams from n=1 to n=5 and collect F1 scores.

    Args:
        data: The dataset containing 'text' and 'label' columns.
        n_values: Range of n-gram values to train classifiers on.

    Returns:
        lists of tuples containing n and the corresponding F1 score.
    """
    f1_scores_j = []
    f1_scores_c = []

    for n in n_values:
        print(f"\nTraining MultinomialNB with n-grams (n={n})...")

        # Initialize BoWEstimator with n-gram range
        bow_estimator = BoWEstimator(MultinomialNB(), " ".join, n)

        # train
        bow_estimator.fit(data.train.get_samples(), data.train.get_labels())
        y_pred = bow_estimator.predict(data.test.get_samples())
        # Evaluate the model
        f1_j = f1_score(
            data.test.get_labels(),
            y_pred,
            pos_label=0
        )
        f1_c = f1_score(data.test.get_labels(), y_pred, pos_label=1)
        f1_scores_j.append((n, f1_j))
        f1_scores_c.append((n, f1_c))
        print(f"F1 Score for n={n}: {f1_j:.4f}, {f1_c:.4f}")
        print(f"Classification Report for n={n}: ")
        print(classification_report(data.test.get_labels(), y_pred))

    return f1_scores_j, f1_scores_c


def plot_f1_scores(f1_scores_j: list[tuple[int, float]], f1_scores_c: list[tuple[int, float]]) -> None:
    """Plot F1 scores for each n.

    Args:
        f1_scores (list[tuple[int, float]]): A list of tuples containing n and the corresponding F1 score.
    """
    # Extract n values and F1 scores
    n_values = [n for n, _ in f1_scores_j]
    scores = [score for _, score in f1_scores_j]

    # Plot the F1 scores
    plt.plot(
        n_values,
        scores,
        marker="o",
        linestyle="-",
        color="blue",
        label="Jewish",
    )
    n_values = [n for n, _ in f1_scores_c]
    scores = [score for _, score in f1_scores_c]
    plt.plot(
        n_values,
        scores,
        marker="o",
        linestyle="-",
        color="green",
        label="Christian",
    )
    plt.title("F1 Scores for MultinomialNB with Different n-grams", fontsize=14)
    plt.xlabel("n (n-gram value)", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.xticks(n_values)
    plt.ylim(0.0, 1.0)
    plt.grid(visible=True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Load your dataset
    # Example: Assuming the dataset has 'text' and 'label' columns
    data = load_etcbc_dataset()

    # Train MultinomialNB classifiers with n-grams from n=1 to n=5
    f1_scores_j, f1_scores_c = train_multinomial_nb_with_ngrams(data, range(1, 6))

    # Plot the F1 scores
    plot_f1_scores(f1_scores_j, f1_scores_c)
