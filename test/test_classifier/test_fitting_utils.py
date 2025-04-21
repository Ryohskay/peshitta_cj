from collections import Counter

from nltk.util import ngrams

from src.classifier.fitting_utils import (
    count_n_grams,
    identity,
    make_feature,
    make_vocab,
    vectorise,
)
from src.classifier.result_utils import PeshittaWord


def test_count_n_grams(cal_words: list[PeshittaWord]):
    words = [w.translit for w in cal_words]
    result = count_n_grams(words, identity, span=2)
    expected = Counter(ngrams(words, 2))
    assert result == expected


def test_make_vocab(
    cal_words: list[PeshittaWord], etcbc_words: list[PeshittaWord]
):
    # test for word n-gram
    n_span = 2
    verses = [
        [w.translit for w in cal_words],
        [w.translit for w in etcbc_words],
    ]
    vocabs = make_vocab(verses, identity, span=2)
    assert len(vocabs) == 2
    counters = [
        Counter(ngrams(verses[0], n_span)),
        Counter(ngrams(verses[1], n_span)),
    ]
    res = Counter(ngrams(verses[0], n_span))
    res.update(Counter(ngrams(verses[1], n_span)))
    assert vocabs[0] == res
    assert vocabs[1] == counters
    # test for char n-gram
    char_verses = [verses[0][:2], verses[1][:2]]
    char_counters = Counter(ngrams(" ".join(char_verses[0]), n_span))
    char_counters.update(Counter(ngrams(" ".join(char_verses[1]), n_span)))

    vocabs = make_vocab(char_verses, " ".join, span=n_span)
    assert len(vocabs) == 2
    assert vocabs[0] == char_counters
    assert vocabs[1] == [
        Counter(ngrams("br$yt br)", n_span)),
        Counter(ngrams("BRCJT BR>", n_span)),
    ]


def test_make_feature():
    n_gram_vocabs = (
        Counter({("this", "is"): 1, ("is", "a"): 1, ("a", "test"): 1}),
        [
            Counter({("this", "is"): 1, ("is", "a"): 1}),
            Counter({("a", "test"): 1}),
        ],
    )
    features = make_feature(n_gram_vocabs)
    expected = [[1, 1, 0], [0, 0, 1]]
    assert features == expected


def test_vectorise():
    verses = [["this", "is", "a", "test"], ["another", "test", "case"]]
    vocab = Counter(
        {
            ("this", "is"): 1,
            ("is", "a"): 1,
            ("a", "test"): 1,
            ("another", "test"): 1,
            ("test", "case"): 1,
        }
    )
    ngram_formatter = identity
    span = 2
    result = vectorise(verses, vocab, ngram_formatter, span)
    expected = [[1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]
    assert result == expected


def test_identity():
    assert identity("test\u0308") == "test\u0308"
    assert identity(123) == 123
    assert identity([1, 2, 3]) == [1, 2, 3]
    assert identity(None) is None
