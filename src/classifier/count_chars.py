from collections import Counter

from classifier.textfabric_utils import load_etcbc_dataset
from classifier.result_utils import Verse


def find_top_k_chars(verses: list[Verse]) -> None:
    translit_chars = []
    syr_chars = []

    for v in verses:
        v_words = v.get_translit_words()
        syr_chars.extend([char for w in v.get_syriac_words() for char in w])
        translit_chars.extend([char for w in v_words for char in w])

    translit_char_counts = Counter(translit_chars)
    print("Most common chars: "
          + f"{translit_char_counts.most_common()}")

    syr_char_counts = Counter(syr_chars)
    print("Most common chars: "
          + f"{syr_char_counts.most_common()}")

    num_chars = syr_char_counts.total()
    percents = [(char[0], (char[1] / num_chars), char[1])
                for char in syr_char_counts.most_common()]

    for p in percents:
        print(f"{p[0]}: {p[1] * 100:.01f} % ({p[2]})")


if __name__ == "__main__":
    k = 150
    etc_load = load_etcbc_dataset()

    print("ETCBC")
    print("Overall - train")
    find_top_k_chars(etc_load.train.get_samples())

    print("OT - train")
    find_top_k_chars(etc_load.train.get_samples(0))

    print("NT - train")
    find_top_k_chars(etc_load.train.get_samples(1))

    print("Overall - test")
    find_top_k_chars(etc_load.test.get_samples())

    print("OT - test")
    find_top_k_chars(etc_load.test.get_samples(0))

    print("NT - test")
    find_top_k_chars(etc_load.test.get_samples(1))
