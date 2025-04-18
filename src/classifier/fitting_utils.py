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

"""Utility functions to handle learning algorithms."""

from collections import Counter
from collections.abc import Callable, Sequence

from nltk.util import ngrams

from src.classifier.result_utils import Verse


def count_n_grams(
    words: list[str], ngram_formatter: Callable, span: int = 3
) -> Counter:
    """Count n-grams in a given verse, using nltk.util.ngrams().

    Args:
        words: list of strings to feed into :func:`nltk.util.ngrams()`.
        ngram_formatter: This is a pre-processing function or method to apply
            to the text of each verse, before passing it to
            :func:`nltk.util.ngrams()`.
            The format of strings passed to ngrams() function determines if
            the function returns a list of character n-grams or that of word
            n-grams. If you pass a single string, it returns character n-grams,
            and if you pass a list of strings (= words), it returns
            word n-grams.
        span: the size of window for n-gram extraction, i.e. *N* of n-grams.

    Returns:
        A :class:Counter object, containing a key-value pair of
        { n-gram: count }. This is basically a dict[str, int] object, but it
        returns **0** if you access a non-existent n-gram.
    """
    # Format the verse to feed into ngrams()
    formatted_inputs = ngram_formatter(words)

    # Generate and count ngrams
    n_grams = list(ngrams(formatted_inputs, span))
    return Counter(n_grams)


def make_vocab(
    verses: list[list[str]],
    ngram_formatter: Callable,
    span: int = 3
) -> tuple[Counter, list[Counter]]:
    """Construct the model's vocabulary by extracting n-grams from verses.

    Args:
        verses: list of verses as str to find n-grams.
        ngram_formatter: This is a pre-processing function or method to apply
            to the text on each verse.
            See :func:`src.classifier.fitting_utils.count_n_grams` for more details.
        span: the size of window for n-gram extraction, i.e. *N* of n-grams.

    .. seealso::
        :meth:`src.classifier.result_utils.Verse.get_words_in_mode`
            for param: ``mode``.

    Returns:
        A tuple in the format of::

            tuple[(Global n-gram counts), (n-gram counts for each verse)]

    Raises:
        ValueError: if ``verses`` has 0 verse in it.
        RuntimeError: if n_gram_vocabs is None even after the parsing is done.
    """
    n_gram_vocabs = None
    n_gram_counters = []
    word_cnt = 0

    if len(verses) < 1:
        msg = "Provided verse seems empty. Please check the arguments."
        raise ValueError(msg)

    # Parse the verses
    for verse in verses:
        word_cnt += len(verse)

        # count n-grams
        n_gram_counter = count_n_grams(
                                        verse,
                                        ngram_formatter,
                                        span
                                    )

        n_gram_counters.append(n_gram_counter)

        if n_gram_vocabs is None:
            n_gram_vocabs = n_gram_counter.copy()
        else:
            n_gram_vocabs.update(n_gram_counter)

    print(f"Total words parsed: {word_cnt}")

    if n_gram_vocabs is None:
        msg = ("Verse parsing completed but the n-gram counter is empty. "
                + "Make sure that the arguments are properly defined.")
        raise RuntimeError(msg)

    return (n_gram_vocabs, n_gram_counters)


def identity(input_data: type) -> type:
    """Identity function that returns the same thing as the input.

    Args:
        input_data: any value of any type.

    Returns:
        Identical to the input.
    """
    return input_data


def make_word_n_gram_vocab(
    verses: list[Verse], span: int = 3, mode: int = 1
) -> tuple[Counter, list[Counter]]:
    """Wraps the ``make_vocab`` function with a formatter for word n-grams.

    .. seealso:
        See :func:`src.classifier.fitting_utils.make_vocab`
            for params: ``verses``, ``span``, ``mode``.
        See :meth:`src.classifier.result_utils.Verse.get_words_in_mode`
            for param: ``mode``.

    Returns:
        A tuple in the format of::

            tuple[(Global word n-gram counts),
                    (word n-gram counts for each verse)]
    """
    verses_s = [v.get_words_in_mode(mode) for v in verses]
    return make_vocab(verses_s, identity, span, mode)


def make_char_n_gram_vocab(
        verses: list[Verse], span: int = 3, mode: int = 1,
        ngram_formatter: Callable = " ".join
) -> tuple[Counter, list[Counter]]:
    """Wraps the ``make_vocab`` function with a formatter for character n-grams.

    .. seealso:
        See :func:`src.classifier.fitting_utils.make_vocab`
            for params: ``verses``, ``span``, ``mode``, ``ngram_formatter``.

    Returns:
        A tuple in the format of::

            tuple[(Global character n-gram counts),
                    (character n-gram counts for each verse)]
    """
    verses_s = [v.get_words_in_mode(mode) for v in verses]
    return make_vocab(verses_s, ngram_formatter, span, mode)


def make_feature(
    n_gram_vocabs: tuple[Counter, list[Counter]],
) -> Sequence[Sequence[int]]:
    """Make a feature vector from the list of pre-calculated Counter objects.

    This function leverages the :class:`Counter` objects generated when
    constructing the model n-gram vocabulary in
    :func:``src.classifier.fitting_utils.make_vocab``

    Args:
        n_gram_vocabs: return values of
            :func:``src.classifier.fitting_utils.make_vocab``

    Returns:
        A two-dimensional list containing feature vectors representing
        the input verses by Bag-of-Words strategy.
    """
    feature_list = []
    for i in range(len(n_gram_vocabs[1])):
        # Make a skeleton dict to use as the BoW feature, set all counts to 0
        n_gram_bow = _make_empty_bow(n_gram_vocabs[0])
        # Add counts to words that exist in this verse
        n_gram_bow.update(n_gram_vocabs[1][i])
        verse_features = list(n_gram_bow.values())
        feature_list.append(verse_features)
    return feature_list


def _make_empty_bow(vocabulary: dict) -> dict[tuple[str], int]:
    # Construct an empty Bag of Words from a Counter or dict object,
    # whose keys are the n-grams.
    return dict.fromkeys(vocabulary.keys(), 0)


def merge_counts(counter: Counter, bow: dict) -> None:
    """Update the value of feat_vec with the value found in Counter.

    This function makes sure that the length of the feature vector
    doesn't change when updating the n-gram counts in feat_vec, even with
    unseen input vocabulary.
    If you use dict.update(), you might accidentally add extra vocabs to
    feat_vec (which changes the length of the feature vector).


    Args:
        counter: :class:`Counter` instance containing counts for n-grams
            in a particular verse.
        bow: empty or populated Bag-of-Words in a dictionary-like format,
            containing the model's predefined n-gram vocabulary and n-gram
            counts.
    """
    for n_gram in counter:
        if n_gram in bow:
            bow[n_gram] = counter[n_gram]


def vectorise(
        verses: list[list[str]],
        vocab: Counter,
        ngram_formatter: Callable,
        span: int = 3
    ) -> list[list[int]]:
    """Count n-grams in unseen verses, using predefined vocabulary.

    Args:
        verses: list of verses to find n-grams and vectorise for further
            processing.
        vocab: empty or populated Bag-of-Words in a dictionary-like format,
            containing the model's predefined n-gram vocabulary and n-gram
            counts.
        ngram_formatter: This is a pre-processing function or method to apply
            to the text on each verse.
            See :func:`src.classifier.fitting_utils.count_n_grams`
            for more details.
        span: the size of window for n-gram extraction, i.e. *N* of n-grams.

    .. seealso::
        :meth:`src.classifier.result_utils.Verse.get_words_in_mode`
            for param: ``mode``.

    Returns:
        A two-dimensional list containing feature vectors representing
        the input verses by Bag-of-Words strategy.
        Thus, generates a list: [N_GRAM_COUNTS per each verse]
    """
    wc = 0
    verse_n_grams = []

    bag_of_words = _make_empty_bow(vocab)

    for verse in verses:
        wc += len(verse)
        local_n_gram_counts = count_n_grams(
            words=verse,
            ngram_formatter=ngram_formatter,
            span=span
        )
        n_gram_bow = dict.fromkeys(bag_of_words.keys(), 0)
        merge_counts(local_n_gram_counts, n_gram_bow)
        verse_n_grams.append(list(n_gram_bow.values()))
    print(f"Parsed {wc} words from {len(verses)} verses")
    return verse_n_grams
