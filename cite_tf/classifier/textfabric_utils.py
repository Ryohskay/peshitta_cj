from tf.app import use

from collections.abc import Callable
from collections import Counter
from nltk.util import ngrams

import numpy as np


def get_verses(
        target_books: dict[str, list[int]],
        target_fabric: str="etcbc/peshitta",
        ver: str="0.2"
    ) -> dict[str, list[tuple[str, list[str]]]]:
    """Extract verses from target_fabric.
    
    Return a dict of format: {"BOOK_TITLE": [list of ("VERSE_REF", [list of words])]}
    """
    # Load text-fabric library
    api_handler = use(target_fabric, hoist=globals(), version=ver)
    result_verses = None
    
    for book in Fs("book@en").items():    
        if book[1] in target_books.keys():
            book_verses = []
            print(book[1])
            chapters = L.d(book[0], otype="chapter")
            # results = "Verse Reference,Lemmatised Transliteration,Plain Transliteration,Original Syriac Text\n"
            for chapter in chapters:
                if int(F.chapter.v(chapter)) in target_books[book[1]]:
                    for verse in L.d(chapter, otype="verse"):
                        verse_ref = f"{book[1]} Chapter {int(F.chapter.v(chapter)):02} Verse {int(F.verse.v(verse)):02}"
                        # get all words in this verse
                        words = L.d(verse, otype="word")
                        # transliteration of this verse as a list of words
                        translit_verse = [F.word_etcbc.v(w_id) for w_id in words]
                        # original Syriac text of this verse, as a list of words
                        syriac_verse = [F.word.v(w_id) for w_id in words]
                        book_verses.append((verse_ref, translit_verse, syriac_verse))
            if result_verses is None:
                result_verses = {book[1]: book_verses}
            else:
                result_verses.update({book[1]: book_verses})
    return result_verses


def count_n_grams(
        word_count: int,
        verse_words: list[str],
        ngram_formatter: Callable,
        span: int=3
    ) -> (int, Counter):
    
    # Format the verse to feed into ngrams()
    formatted_inputs = ngram_formatter(verse_words)
    
    # Generate and count ngrams
    n_grams = list(ngrams(formatted_inputs, span))
    n_gram_counts = Counter(n_grams)
    
    # Keep the ngram counters
    return (word_count+len(verse_words), n_gram_counts)


def make_vocab(verses: list[tuple[str, list[str]]], ngram_formatter: Callable, span: int=3) -> tuple[Counter, list[Counter]]:
    word_cnt = 0
    n_gram_vocabs = None
    n_gram_counters = []

    for verse in verses:
        word_cnt, n_gram_counter = count_n_grams(word_cnt, verse[1], ngram_formatter, span)

        n_gram_counters.append(n_gram_counter)
    
        if n_gram_vocabs is None:
            n_gram_vocabs = n_gram_counter.copy()
        else:
            n_gram_vocabs.update(n_gram_counter)

    print(f"Total words parsed: {word_cnt}")
    return (n_gram_vocabs, n_gram_counters)


def identity(input: type) -> type:
    return input


def make_word_n_gram_vocab(verses: list[tuple[str, list[str]]], span: int=3) -> tuple[Counter, list[Counter]]:
    return make_vocab(verses, identity, span)


def make_char_n_gram_vocab(verses: list[tuple[str, list[str]]], span: int=3) -> tuple[Counter, list[Counter]]:
    return make_vocab(verses, ' '.join, span)


def make_bow(vocabulary: dict) -> dict[tuple[str], int]:
    return dict.fromkeys(vocabulary.keys(), 0)


def make_feature(n_gram_vocabs: tuple[Counter, list[Counter]]) ->list[list[int]]:
    feature_list = []
    for i in range(len(n_gram_vocabs[1])):
        # Make a skeleton dict to use as the BoW feature, set all counts to 0
        n_gram_bow = make_bow(n_gram_vocabs[0])
        # Add counts to words that ecist in this verse
        n_gram_bow.update(n_gram_vocabs[1][i])
        verse_features = list(n_gram_bow.values())
        feature_list.append(verse_features)
    return feature_list


def merge_counts(counter: Counter, feat_vec: dict) -> None:
    for n_gram in counter.keys():
        # print(n_gram)
        if n_gram in feat_vec.keys():
            feat_vec[n_gram] = counter[n_gram]


def vectorise(
        verses: list[tuple[str, list[str]]],
        bag_of_words: Counter,
        ngram_formatter: Callable,
        span: int=3
    ) -> list[list[int]]:
    # Get the verse of format [(VERSE_REF, [VERSE_translit_WORDS], [VERSE_syriac_WORDS])]
    # Generate two lists: [[N_GRAM_COUNTS] per each verse] & [LABEL per each verse]
    labels = []
    wc = 0
    verse_n_grams = []
    for verse in verses:
        wc, local_n_gram_counts = count_n_grams(word_count=wc, verse_words=verse[1], ngram_formatter=ngram_formatter, span=3)
        n_gram_bow = dict.fromkeys(bag_of_words.keys(), 0)
        merge_counts(local_n_gram_counts, n_gram_bow)
        verse_n_grams.append(list(n_gram_bow.values()))
    print(f"Parsed {wc} words from {len(verses)} verses")
    return verse_n_grams


def predict(
        classifier, test_X: np.ndarray, test_labels: np.array
    ) -> (np.array, np.array, np.array):
    """Predict on the data with a given classifier, and return some simple statistics.

    The classifier must have a method `.predict()`.
    """
    y_pred = classifier.predict(test_X)
    
    num_samples = test_X.shape[0]
    num_mislabels = ((test_labels != y_pred).sum())
    num_correct = num_samples - num_mislabels
    
    print("Number of mislabeled points out of a total %d OT verses: %d" % (num_samples, num_mislabels))
    print(f"Local accuracy: {(num_correct/num_samples):.02f}")
    return y_pred, num_mislabels, num_correct


def csvify(verses: list[tuple[str, list[str]]]) -> str:
    """Convert the verse data into a CSV-formatted string."""
    # Set header line
    result = '"Verse Reference No.","ܐܠܦ ܒܝܬ ܣܘܪܝܝܐ","ETCBC Transliteration"\n'
    # Extract & format verse data
    for verse in verses:
        line = f'"{verse[0]}","{' '.join(verse[2])}","{' '.join(verse[1])}"\n'
        result += line
    return result
