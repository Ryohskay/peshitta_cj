from pathlib import Path

import pandas as pd

from classifier import book_data


def normalise_title(title: str) -> str:
    """Normalise the title of biblical books according to ETCBC."""
    if "_" in title:
        # 1_Samuel to Samuel_1
        return "_".join(title.split("_")[::-1])
    if "Nehemiah":
        return "Nehemia"
    return title


def is_ot_book(title: str) -> bool:
    """Check if a particular book is in the OT or NT."""
    book_title = normalise_title(title)
    if (
            book_title in book_data.ot_train_books.keys()
            or book_title in book_data.ot_test_books.keys()
            or book_title in book_data.ot_prod_books.keys()
    ):
        return True
    return False


def load_df_json(data_dir: str | Path) -> pd.DataFrame:
    """Get CAL scraper results in json format and load as pd.DataFrame."""
    total_df = None
    for fpath in Path(data_dir).iterdir():
        # Load the json data
        df = pd.read_json(fpath)
        # Append to total_df
        if total_df is None:
            total_df = df
        else:
            total_df = pd.concat([total_df, df], ignore_index=True)
    return total_df


def get_book(df: pd.DataFrame, title: str) -> pd.DataFrame:
    """Filter the DataFrame by book_title column."""
    return df.loc[df["book_title"] == title]


def verse_in_chapters(verse: tuple, chapters: list[int]):
    return int(verse.verse_refs.split(" ")[2]) in chapters


def get_book_verses(
        df: pd.DataFrame,
        target_books: dict,
        trim_none: bool=False
    ) -> list:
    """Get all books in the dataframe and generate a list of tuples.

    Returns a list representing each verse as a tuple.
    Each tuple contains: [0] verse reference, [1] list of lemmata in the verse,
    and [2] list of tuples for each lemma with its annotations (lemma, annots).
    """
    books = None
    verse_box = []

    load_books = target_books.keys()
    for bk in load_books:
        print(bk)
        book_df = get_book(df, bk)
        if books is None:
            books = book_df
        else:
            books = pd.concat([books, book_df], ignore_index=True)

        selected_cols = books[["verse_refs", "lemmatised_verses", "lemma_annotations"]]
        for verse in selected_cols.itertuples():
            if verse_in_chapters(verse, target_books[bk]):
                refs = verse.verse_refs
                lemmata = []        # list[lemma]
                lemma_annots = []   # list[(lemma, annotations)]
                for i in range(len(verse.lemmatised_verses)):
                    lemma = verse.lemmatised_verses[i]
                    annot = verse.lemma_annotations[i]
                    annot_r = annot if annot is not None else ""
                    # Handle lemma
                    if (trim_none and lemma is not None) or not trim_none:
                        lemmata.append(lemma)
                    else:
                        continue
                    lemma_annots.append((lemma, annot_r))
                verse_box.append((refs, lemmata, lemma_annots))
    return verse_box


if __name__ == "__main__":
    df = load_df_json("../scraper/cal_results/")
    print(df.columns)

    # Parse the DF
    print(get_book(df, "Acts"))

    ot_train = get_book_verses(df, book_data.ot_train_books)
    ot_test = get_book_verses(df, book_data.ot_test_books)
    ot_prod = get_book_verses(df, book_data.ot_prod_books)
    nt_train = get_book_verses(df, book_data.nt_train_books)
    nt_test = get_book_verses(df, book_data.nt_test_books)
    print(ot_train)
    print(ot_test)
    print(nt_train)
