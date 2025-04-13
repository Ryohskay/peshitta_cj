"""Utility functions to sanitise str."""


def normalise_book_title(book_title: str) -> str:
    """Normalises a book title to align with the text-fabric names.

    Returns:
        A str containing the book title after normalisation.
    """
    # Mapping of book titles where ETCBC and CAL diverge.
    etcbc_cal_map = {
                "1_Chronicles": "Chronicles_1",
                "1_Samuel": "Samuel_1",
                "2_Samuel": "Samuel_2",
                "1_Kings": "Kings_1",
                "2_Kings": "Kings_2",
                "Nehemiah": "Nehemia",
                "2_Chronicles": "Chronicles_2",
                "1_Maccabees": "Maccabees_1_B",
            }
    if book_title in etcbc_cal_map:
        return etcbc_cal_map[book_title]
    # else
    return book_title.title()


def sanitise_str(s: str) -> str:
    """Returns a sanitised string.

    This function performs lower-casing and stripping off non-word characters.
    """
    return s.lower().strip()


def clean_path_str(s: str) -> str:
    """Performs sanitisation on a str treated as a path.

    This function removes a prefixed slash to prevent accidental file access to
    the root directory on *nix systems (it will probably raise some OSError).
    It also strips off non-word characters.

    Returns:
        A str containing a sanitised path.
    """
    return s.removeprefix("/").strip()
