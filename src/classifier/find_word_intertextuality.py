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

from collections import Counter

from classifier.fitting_utils import make_vocab
from classifier.inspection_utils import find_top_k_words
from classifier.load_cal import load_cal_dataset
from classifier.textfabric_utils import load_etcbc_dataset

if __name__ == "__main__":
    k = 150
    etc_load = load_etcbc_dataset()
    translit_words = []
    translit_chars = []

    syr_chars = []

    print("ETCBC")
    for v in etc_load.train.get_samples():
        v_words = v.get_translit_words()
        syr_chars.extend([char for w in v.get_syriac_words() for char in w])
        translit_words.extend(v_words)
        translit_chars.extend([char for w in v_words for char in w])

    print("Most common chars: "
          + f"{Counter(syr_chars).most_common()}")

    print("Most common chars: "
          + f"{Counter(syr_chars).most_common()}")

    print(f"\nTop {k} common words: "
          + f"{Counter(translit_words).most_common(k)}")

    print("\n>> ETCBC/peshitta_OT")
    for v in etc_load.train.get_samples(0):
        v_words = v.get_translit_words()
        translit_words.extend(v_words)
        translit_chars.extend([char for w in v_words for char in w])

    print("Most common chars: "
          + f"{Counter(translit_chars).most_common()}")
    print(f"\nTop {k} common words: "
          + f"{Counter(translit_words).most_common(k)}")

    print("\n>> ETCBC/syrNT")
    for v in etc_load.train.get_samples(1):
        v_words = v.get_translit_words()
        translit_words.extend(v_words)
        translit_chars.extend([char for w in v_words for char in w])

    print("Most common chars: "
          + f"{Counter(translit_chars).most_common()}")
    print(f"Top {k} common words: "
          + f"{Counter(translit_words).most_common(k)}")

    print("\nCAL")
    cal_load = load_cal_dataset()
    for v in cal_load.train.get_samples():
        v_words = v.get_translit_words()
        translit_words.extend(v_words)
        translit_chars.extend([char for w in v_words for char in w])

    char_counts = Counter(translit_chars)
    print("Most common chars: "
          + f"{char_counts}")

    num_chars = char_counts.total()
    percents = [(char[0], (char[1] / num_chars), char[1])
                for char in char_counts.most_common()]

    for p in percents:
        print(f"{p[0]}: {p[1] * 100:.01f} % ({p[2]})")

    print(f"\nTop {k} common words: "
          + f"{Counter(translit_words).most_common(k)}")
    print(f"\nTop {k} common words: "
          + f"{Counter(translit_words).most_common(k)}")

    print("\n>> CAL/OT")
    for v in cal_load.train.get_samples(0):
        v_words = v.get_translit_words()
        translit_words.extend(v_words)
        translit_chars.extend([char for w in v_words for char in w])

    char_counts = Counter(translit_chars)
    print("Most common chars: "
          + f"{char_counts}")

    num_chars = char_counts.total()
    percents = [(char[0], (char[1] / num_chars), char[1])
                for char in char_counts.most_common()]

    for p in percents:
        print(f"{p[0]}: {p[1] * 100:.01f} % ({p[2]})")
    print(f"\nTop {k} common words: "
          + f"{Counter(translit_words).most_common(k)}")

    print("\n>> CAL/NT")
    for v in cal_load.train.get_samples(1):
        v_words = v.get_translit_words()
        translit_words.extend(v_words)
        translit_chars.extend([char for w in v_words for char in w])

    char_counts = Counter(translit_chars)

    num_chars = char_counts.total()
    percents = [(char[0], (char[1] / num_chars), char[1])
                for char in char_counts.most_common()]

    for p in percents:
        print(f"{p[0]}: {p[1] * 100:.01f} % ({p[2]})")

    print(f"\nTop {k} common words: "
          + f"{Counter(translit_words).most_common(k)}")

    n = 3
    ngram_counts, vocabs = make_vocab(
            etc_load.train.get_samples(), " ".join, span=n)
    ngrams = ngram_counts.most_common(k)
    print(f"Top {k} common {n}-grams: "
          + f"{[''.join(list(ngram[0])) for ngram in ngrams]}")
