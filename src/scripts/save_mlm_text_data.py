import json

from pathlib import Path
from typing import TypedDict
from src.classifier.textfabric_utils import get_verses
from src.classifier.result_utils import Verse

OT_TARGETS = {
    # BOOK: CHAPTER NO.S
    # Since the current impl picks up every verse and compares against the list,
    # set the chapters to 1..100 to pick up all verses
    '3Esdras': list(range(1, 100)),
    '3Maccabees': list(range(1, 100)),
    '4Esdras': list(range(1, 100)),
    '4Maccabees': list(range(1, 100)),
    'Amos': list(range(1, 100)),
    'Apocalypse_of_Baruch': list(range(1, 100)),
    'Apocryphal_Psalms': list(range(1, 100)),
    'Apocryphal_Psalms_A': list(range(1, 100)),
    'Apocryphal_Psalms_B': list(range(1, 100)),
    'Baruch': list(range(1, 100)),
    'Bel_and_the_Dragon': list(range(1, 100)),
    'Daniel': list(range(1, 100)),
    'Ecclesiastes': list(range(1, 100)),
    'Ezekiel': list(range(1, 100)),
    'Habakkuk': list(range(1, 100)),
    'Haggai': list(range(1, 100)),
    'Hosea': list(range(1, 100)),
    'Isaiah': list(range(1, 100)),
    'Jeremiah': list(range(1, 100)),
    'Job': list(range(1, 100)),
    'Joel': list(range(1, 100)),
    'Jonah': list(range(1, 100)),
    'Judith': list(range(1, 100)),
    'Lamentations': list(range(1, 100)),
    'Letter_of_Baruch_A': list(range(1, 100)),
    'Letter_of_Baruch_B': list(range(1, 100)),
    'Letter_of_Jeremiah': list(range(1, 100)),
    'Leviticus': list(range(1, 100)),
    'Malachi': list(range(1, 100)),
    'Micah': list(range(1, 100)),
    'Nahum': list(range(1, 100)),
    'Numbers': list(range(1, 100)),
    'Obadiah': list(range(1, 100)),
    'Odes': list(range(1, 100)),
    'Prayer_of_Manasseh_A': list(range(1, 100)),
    'Prayer_of_Manasseh_B': list(range(1, 100)),
    'Proverbs': list(range(1, 100)),
    'Psalms': list(range(1, 100)),
    'Psalms_of_Solomon': list(range(1, 100)),
    'Sirach': list(range(1, 100)),
    'Song_of_Songs': list(range(1, 100)),
    'Susanna': list(range(1, 100)),
    'Tobit_A': list(range(1, 100)),
    'Tobit_B': list(range(1, 100)),
    'Wisdom_of_Solomon': list(range(1, 100)),
    'Zechariah': list(range(1, 100)),
    'Zephaniah': list(range(1, 100))
}

NT_TARGETS = {
    '1_Corinthians': list(range(1, 100)),
    '1_John': list(range(1, 100)),
    '1_Peter': list(range(1, 100)),
    '1_Thessalonians': list(range(1, 100)),
    '1_Timothy': list(range(1, 100)),
    '2_Corinthians': list(range(1, 100)),
    '2_John': list(range(1, 100)),
    '2_Peter': list(range(1, 100)),
    '2_Thessalonians': list(range(1, 100)),
    '2_Timothy': list(range(1, 100)),
    '3_John': list(range(1, 100)),
    'Colossians': list(range(1, 100)),
    'Ephesians': list(range(1, 100)),
    'Galatians': list(range(1, 100)),
    'Hebrews': list(range(1, 100)),
    'James': list(range(1, 100)),
    'Jude': list(range(1, 100)),
    'Philemon': list(range(1, 100)),
    'Philippians': list(range(1, 100)),
    'Revelation': list(range(1, 100)),
    'Romans': list(range(1, 100)),
    'Titus': list(range(1, 100))
}

def to_books_data(verses: list[Verse]) -> dict[str, list[Verse]]:
    data = {}
    for v in verses:
        if v.book not in data:
            data[v.book] = [v]
        else:
            data[v.book].append(v)
    return data


if __name__ == "__main__":
    ot_verses = to_books_data(get_verses(OT_TARGETS))
    nt_verses = to_books_data(get_verses(NT_TARGETS))
    print("OT")
    print(f"{len(ot_verses)} books")
    with Path("src/neural/ot_mlm_data.json").open(encoding="utf-8") as fp:
        json.dump(ot_verses, fp)
    
    print("NT")
    print(f"{len(nt_verses)} books")
    with Path("src/neural/nt_mlm_data.json").open(encoding="utf-8") as fp:
        json.dump(nt_verses, fp)
