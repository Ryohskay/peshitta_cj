"""Utility functions to train algorithms."""

from collections import Counter
from collections.abc import Callable

import numpy as np
from nltk.util import ngrams
from sklearn.base import BaseEstimator


def count_n_grams(
    words: list[str], ngram_formatter: Callable, span: int = 3
) -> Counter:
    """Count n-grams in the verse_words, using nltk.util.ngrams().

    ngram_formatter: Callable
        This is a pre-processing function or method to apply to the text of each verse,
        before passing it to nltk.util.ngrams().

        The format of strings passed to ngrams() function determines if the function
        returns a list of character n-grams or that of word n-grams.

        If you pass a single string, it returns character n-grams, and if you pass
        a list of strings (= words), it returns word n-grams.
    """
    # Format the verse to feed into ngrams()
    formatted_inputs = ngram_formatter(words)

    # Generate and count ngrams
    n_grams = list(ngrams(formatted_inputs, span))
    n_gram_counts = Counter(n_grams)

    # Keep the ngram counters
    return n_gram_counts


def make_vocab(
    verses: list[tuple], ngram_formatter: Callable, span: int = 3, mode: int = 1
) -> tuple[Counter, list[Counter]]:
    """Construct the model's vocabulary by extracting n-grams from verses.

    ngram_formatter: Callable
        This is a pre-processing function or method to apply to the text of each verse.
        See count_n_grams for more details.

    mode: int
        Access an item in verse tuple at this index.
        On ETCBC data:
            if 1, then use ETCBC transliteration
            if 2, then use original Syriac script
        On CAL data:
            must be 1.
    """
    word_cnt = 0
    n_gram_vocabs = None
    n_gram_counters = []

    for verse in verses:
        n_gram_counter = count_n_grams(verse[mode], ngram_formatter, span)
        word_cnt += len(verse[mode])

        n_gram_counters.append(n_gram_counter)

        if n_gram_vocabs is None:
            n_gram_vocabs = n_gram_counter.copy()
        else:
            n_gram_vocabs.update(n_gram_counter)

    print(f"Total words parsed: {word_cnt}")
    return (n_gram_vocabs, n_gram_counters)


def identity(input: type) -> type:
    return input


def make_word_n_gram_vocab(
    verses: list[tuple], span: int = 3, mode: int = 1
) -> tuple[Counter, list[Counter]]:
    return make_vocab(verses, identity, span, mode)


def make_char_n_gram_vocab(
    verses: list[tuple], span: int = 3, mode: int = 1
) -> tuple[Counter, list[Counter]]:
    return make_vocab(verses, " ".join, span, mode)


def make_feature(
    n_gram_vocabs: tuple[Counter, list[Counter]],
) -> list[list[int]]:
    feature_list = []
    for i in range(len(n_gram_vocabs[1])):
        # Make a skeleton dict to use as the BoW feature, set all counts to 0
        n_gram_bow = _make_empty_bow(n_gram_vocabs[0])
        # Add counts to words that ecist in this verse
        n_gram_bow.update(n_gram_vocabs[1][i])
        verse_features = list(n_gram_bow.values())
        feature_list.append(verse_features)
    return feature_list


def _make_empty_bow(vocabulary: dict) -> dict[tuple[str], int]:
    return dict.fromkeys(vocabulary.keys(), 0)


def merge_counts(counter: Counter, feat_vec: dict) -> None:
    """Update the value of feat_vec with the value found in Counter.

    This function makes sure that the length of the feature vector
    doesn't change when updating the n-gram counts in feat_vec.
    If you use dict.update(), you accidentally add extra vocabs to
    feat_vec (which changes the length of the feature vector).
    """
    for n_gram in counter.keys():
        # print(n_gram)
        if n_gram in feat_vec:
            feat_vec[n_gram] = counter[n_gram]


def vectorise(
    verses: list[tuple],
    vocab: Counter,
    ngram_formatter: Callable,
    span: int = 3,
) -> list[list[int]]:
    """Count n-grams in unseen verses, using the vocabulary from bag_of_words."""
    # Get the verse of format [(VERSE_REF, [VERSE_translit_WORDS], [VERSE_syriac_WORDS])]
    # Generate a list: [N_GRAM_COUNTS per each verse]
    wc = 0
    verse_n_grams = []

    bag_of_words = _make_empty_bow(vocab)

    for verse in verses:
        wc += len(verse[1])
        local_n_gram_counts = count_n_grams(
            words=verse[1], ngram_formatter=ngram_formatter, span=span
        )
        n_gram_bow = dict.fromkeys(bag_of_words.keys(), 0)
        merge_counts(local_n_gram_counts, n_gram_bow)
        verse_n_grams.append(list(n_gram_bow.values()))
    print(f"Parsed {wc} words from {len(verses)} verses")
    return verse_n_grams


class BoW_Estimator(BaseEstimator):
    """Wrapper around the BoW vectorisation to allow seamless fitting and predicting."""

    def __init__(
        self,
        clf,
        formatter,
        n: int = 3,
    ):
        self.algo = clf
        self.n = n
        self.n_gram_formatter = formatter
        self.vocabs = None
        self.train_x = None
        self.pred_x = None

    def fit(self, X, y, sample_weight=None):
        if len(X) != len(y):
            msg = f"Length of X ({len(X)}) and y ({len(y)}) do not match."
            raise ValueError(msg)
        self.vocabs = make_char_n_gram_vocab(X, span=self.n)
        print(
            f"Found {len(self.vocabs[0])} independent n-grams "
            + f"from {len(self.vocabs[1])} verses!"
        )
        n_gram_feat = make_feature(self.vocabs)
        self.train_x = n_gram_feat
        self.algo.fit(n_gram_feat, y, sample_weight)

    def predict(self, X):
        targets = np.array(
            vectorise(X, self.vocabs[0], self.n_gram_formatter, span=self.n)
        )
        self.pred_x = targets
        return self.algo.predict(targets)

    def predict_proba(self, X):
        targets = np.array(
            vectorise(X, self.vocabs[0], self.n_gram_formatter, span=self.n)
        )
        return self.algo.predict_proba(targets)

    def get_metadata_routing(self):
        return self.algo.get_metadata_routing()

    def get_params(self, deep=True):
        return self.algo.get_params(deep)
