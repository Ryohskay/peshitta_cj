"""Count the number of verses, words, and characters in the dataset."""

from collections.abc import Callable
from typing import Literal

from src.classifier.cal_aa_eval import remove_underscores
from src.classifier.dataset_skeleton import LoadedDataset
from src.classifier.etcbc_aa_eval import remove_non_chars
from src.classifier.fitting_utils import identity
from src.classifier.load_cal import load_cal_dataset
from src.classifier.result_utils import Verse
from src.classifier.textfabric_utils import load_etcbc_dataset
from src.shared.label_data import ValToLabel
from src.ui_web.load_predictions import BookVerses
import matplotlib.pyplot as plt
from src.classifier.result_utils import Verse
from src.classifier.dataset_skeleton import LoadedDataset

def get_per_book_verses(
    samples: list[Verse],
    *, trim_none: bool = True
) -> list[BookVerses]:
    """Get the verses per book from the samples."""
    lis_book_verses: list[BookVerses] = []
    current_book = ""
    book_verses = []
    is_first_iter = True
    for i in range(len(samples)):
        if current_book != samples[i].book:
            # if we encounter a new book
            if not is_first_iter:
                # once we finish parsing a book,
                lis_book_verses.append(
                    {
                        "book_name": current_book,
                        "verses": book_verses,
                        "verse_probas": [],
                    }
                )
                # reset the book_verses list
                book_verses = []
            # update current_book
            current_book = samples[i].book
            # mark the first iteration
            is_first_iter = False
        if (
            trim_none
            and samples[i] is not None
            and len(samples[i].get_translit_words()) > 0
        ):
            # for every verse, add the verse to the list
            book_verses.append(samples[i])
        elif not trim_none:
            book_verses.append(samples[i])
    # after the loop, add the last book
    lis_book_verses.append(
        {"book_name": current_book, "verses": book_verses, "verse_probas": []}
    )
    return lis_book_verses


def per_label_per_book_verses(
    book_verses: list[BookVerses],
    label: int = 0,
    preprocessor: Callable[[list[str]], list[str]] = identity,
) -> tuple[int, int, int]:
    """Count the number of verses, words, and characters in each book.

    Args:
        book_verses: The list of books and their verses.
        label: The label for the dataset (0 for OT, 1 for NT, -1 for production)
        preprocessor: A preprocessing function to apply to the verses.

    Returns:
        A tuple containing the total number of verses, words, and characters.
    """
    translit_words_per_book = []
    for book in book_verses:
        book_translit_words = [
            preprocessor(verse.get_translit_words()) for verse in book["verses"]
        ]
        translit_words_per_book.append(book_translit_words)
        # print the number of verses and words in each book
        print(f"Book: {book['book_name']}")
        print(f"Number of verses: {len(book['verses'])}")
        print(
            f"Number of words: {sum([len(verse) for verse in book_translit_words])}"
        )
        print(
            f"Number of characters: {sum([len(' '.join(verse)) for verse in book_translit_words])}"
        )

    num_verses = sum(len(book["verses"]) for book in book_verses)
    num_words = sum(
        len(verse.get_translit_words())
        for book in book_verses
        for verse in book["verses"]
    )
    num_characters = sum(
        len(" ".join(preprocessor(verse.get_translit_words())))
        for book in book_verses
        for verse in book["verses"]
    )
    # print the total number of verses, words, and characters
    label_name = "Production" if label == -1 else ValToLabel[label]
    print(f"\n{label_name} Total number of books: {len(book_verses)}")
    print(f"{label_name} Total number of verses: {num_verses}")
    print(f"{label_name} Total number of words: {num_words}")
    print(f"{label_name} Total number of characters: {num_characters}")
    return num_verses, num_words, num_characters


def count_verses_words_characters(
    dataset: LoadedDataset,
    split: Literal["train", "test", "production"],
    preprocessor: Callable[[list[str]], list[str]],
) -> None:
    """Count the number of verses, words, and characters in the dataset.

    Args:
        dataset: The dataset to analyse.
        split: The string for the name of the data split
        preprocessor: a preprocessing function.
    """
    if split in {"train", "test"}:
        ds = getattr(dataset, split)
        print("\n>> OT")
        per_label_per_book_verses(
            get_per_book_verses(ds.get_samples(0)),
            label=0,
            preprocessor=preprocessor,
        )
        print(">> NT")
        per_label_per_book_verses(
            get_per_book_verses(ds.get_samples(1)),
            label=1,
            preprocessor=preprocessor,
        )
    elif split == "production":
        print(">> Production")
        per_label_per_book_verses(
            get_per_book_verses(dataset.production),
            label=-1,
            preprocessor=preprocessor,
        )

def plot_verses_per_book(
    dataset: LoadedDataset,
    title: str = "Number of Verses per Book",
    output_file: str | None = None,
) -> None:
    """Create a bar chart showing the number of verses per book.

    Args:
        dataset: The dataset to analyze.
        title: The title of the chart.
        output_file: If provided, saves the chart to the specified file.

    Raises:
        ValueError: If the split is not one of "train", "test", or "production".
    """
    # Get the dataset split
    ot_books: list[BookVerses] = []
    nt_books: list[BookVerses] = []
    for split in ["train", "test"]:
        ds = getattr(dataset, split)
        ot_books.extend(get_per_book_verses(ds.get_samples(0), trim_none=True))
        nt_books.extend(get_per_book_verses(ds.get_samples(1), trim_none=True))

    # Extract book names and verse counts for OT and NT
    ot_book_names = [book["book_name"] for book in ot_books]
    ot_verse_counts = [len(book["verses"]) for book in ot_books]

    nt_book_names = [book["book_name"] for book in nt_books]
    nt_verse_counts = [len(book["verses"]) for book in nt_books]

    # Create the bar charts
    fig, axes = plt.subplots(3, 1, figsize=(16, 6), sharey=True)

    # OT chart
    axes[0].bar(ot_book_names, ot_verse_counts, color="skyblue")
    axes[0].set_title("OT Books")
    axes[0].set_xlabel("Books")
    axes[0].set_ylabel("Number of Verses")
    axes[0].tick_params(axis="x", rotation=45)

    # NT chart
    axes[1].bar(nt_book_names, nt_verse_counts, color="lightgreen")
    axes[1].set_title("NT Books")
    axes[1].set_xlabel("Books")
    axes[1].tick_params(axis="x", rotation=45)

    # Set the overall title
    fig.suptitle(title)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save or show the chart
    if output_file:
        plt.savefig(output_file)
        print(f"Chart saved to {output_file}")
    else:
        plt.show()

def plot_verses_per_prod_book(
    dataset: LoadedDataset,
    title: str = "Number of Verses per Book",
    output_file: str | None = None,
) -> None:
    """Create a bar chart showing the number of verses per book.

    Args:
        dataset: The dataset to analyze.
        title: The title of the chart.
        output_file: If provided, saves the chart to the specified file.

    Raises:
        ValueError: If the split is not one of "train", "test", or "production".
    """
    # Get the dataset split
    ot_books = []
    nt_books = []
    for split in ["train", "test"]:
        ds = getattr(dataset, split)
        ot_books.extend(get_per_book_verses(ds.get_samples(0), trim_none=True))
        nt_books.extend(get_per_book_verses(ds.get_samples(1), trim_none=True))

    # Extract book names and verse counts for OT and NT
    ot_book_names = [book["book_name"] for book in ot_books]
    ot_verse_counts = [len(book["verses"]) for book in ot_books]

    nt_book_names = [book["book_name"] for book in nt_books]
    nt_verse_counts = [len(book["verses"]) for book in nt_books]

    # Create the bar charts
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # OT chart
    axes[0].bar(ot_book_names, ot_verse_counts, color="skyblue")
    axes[0].set_title("OT Books")
    axes[0].set_xlabel("Books")
    axes[0].set_ylabel("Number of Verses")
    axes[0].tick_params(axis="x", rotation=45)

    # NT chart
    axes[1].bar(nt_book_names, nt_verse_counts, color="lightgreen")
    axes[1].set_title("NT Books")
    axes[1].set_xlabel("Books")
    axes[1].tick_params(axis="x", rotation=45)

    # Set the overall title
    fig.suptitle(title)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save or show the chart
    if output_file:
        plt.savefig(output_file)
        print(f"Chart saved to {output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    # Load the dataset
    print("CAL")
    dataset_cal = load_cal_dataset(src_dir="src/")

    # Count the number of verses, words, and characters
    count_verses_words_characters(dataset_cal, "train", remove_underscores)
    count_verses_words_characters(dataset_cal, "test", remove_underscores)
    count_verses_words_characters(dataset_cal, "production", remove_underscores)

    print("\nETCBC")
    # Load the ETCBC dataset
    dataset_etc = load_etcbc_dataset()
    count_verses_words_characters(dataset_etc, "train", remove_non_chars)
    count_verses_words_characters(dataset_etc, "test", remove_non_chars)
    count_verses_words_characters(dataset_etc, "production", remove_non_chars)
