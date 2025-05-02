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
