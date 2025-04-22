# Format of verse in this project
When treated as a Python object, the data follows the format below:

- The whole or a part of the dataset is represented as a list
- Each verse is representend as a tuple
- Each tuple contains:
    - [0] Verse reference
        - as: "{book title} Chapter {chapter No.} Verse {verse No.}"
    - [1] Verse
        - as a list of words or lemmata

- The contents of each tuple on and after index [2] differs by source:
    - ETCBC:
        - [2] Verse in Syriac alphabets
            - as a list of words
    - CAL:
        - [2] Annotations of each lemma
            - as a tuple of strings: ({lemma}, {annotations})
