"""Dictionaries to specify which chapters from which book to be fetched."""

# Try Fs("book@en").items() to get names of the book included in the text-fabric dataset
ot_train_books = {
    "Genesis": list(range(1,50+1)),
    "Exodus": list(range(1,21+1))
}

ot_test_books = {
    "Deuteronomy": list(range(1,20+1))
}

ot_prod_books = {
    "Joshua": list(range(1,24+1)),
    "Judges": list(range(1,21+1)),
    "Samuel_1": list(range(1,32+1)),
    "Samuel_2": list(range(1,24+1)),
    "Kings_1": list(range(1,22+1)),
    "Kings_2": list(range(1,26+1)),
    "Ezra": list(range(1,48+1)),
    "Nehemia": list(range(1,13+1)),
    "Chronicles_1": list(range(1,32+1)),
    "Chronicles_2": list(range(1,36+1)),
    "Maccabees_1_A": list(range(1,16+1)),
    "Maccabees_1_B": list(range(1,16+1)),
    "Ruth": list(range(1,4+1)),
    "Esther": list(range(1,10+1)),
}

nt_train_books = {
    "Matthew": list(range(1,30+1)),
    "Mark": list(range(1,16+1)),
    "Luke": list(range(1,24+1)),
    "John": list(range(1,21+1)),
}

nt_test_books = {
    "Acts": list(range(1,28+1))
}
