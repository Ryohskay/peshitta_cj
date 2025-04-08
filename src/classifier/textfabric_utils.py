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

"""Utilities to handle the text-fabric datasets.

If you use any python linter, formatter, type-checker, etc., this file will
produce lots of warnings about variables not defined, but **this is expected**.

The underlying library to handle the ETCBC datasets, ``text-fabric``, has
an idiosyncratic interface designed in a C language style, where a whole
file will be imported into any scripts using the library regardless of scopes.

Thus, by declaring ``import use``, we automatically load objects such as ``F``,
along with all of its methods.
"""


from tf.app import use

from classifier.result_utils import Verse


def get_verses(
    target_books: dict[str, list[int]],
    target_fabric: str = "etcbc/peshitta",
    ver: str = "0.2",
) -> list[Verse]:
    """Extract verses from target_fabric.

    Args:
        target_books: a dictionary containing a book title as the key and
            list of chapter numbers to obtain.
        target_fabric: name of the target fabric (corpus), as in the target
            corpus' github repository name. In this project,
            "etcbc/peshitta" or "etcbc/syrnt".
        ver: version string indicating the target fabric (corpus)'s version.

    Returns:
        a dict object pairing a book title to list of verses in format:
        {"BOOK_TITLE": [list of ("VERSE_REF", [list of words])]}
    """
    # Load text-fabric library
    use(target_fabric, hoist=globals(), version=ver)
    result_verses = []

    for book in Fs("book@en").items():
        if book[1] in target_books:
            # extract all books with names found in ``target_books``
            print(book[1])
            chapters = L.d(book[0], otype="chapter")
            for chapter in chapters:
                if int(F.chapter.v(chapter)) in target_books[book[1]]:
                    # walk through all chapters with chapter numbers found in
                    # ``target_books``
                    for verse in L.d(chapter, otype="verse"):
                        verse_ref = (
                                        f"{book[1]} Chapter "
                                        + f"{int(F.chapter.v(chapter)):02} "
                                        + f"Verse {int(F.verse.v(verse)):02}"
                                     )
                        # get all words in this verse
                        words = L.d(verse, otype="word")
                        # transliteration of this verse as a list of words
                        translit_verse = [
                            F.word_etcbc.v(w_id) for w_id in words
                        ]
                        # original Syriac text of this verse, as a list of words
                        syriac_verse = [F.word.v(w_id) for w_id in words]
                        # append this verse's info to the results list
                        result_verses.append(Verse(book[1], verse_ref,
                              translit_verse, syriac_verse,
                              origin="ETCBC"))
    return result_verses
