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

"""Dictionaries to specify which chapters of which book will be fetched.

.. attention::
    The dictionaries in this file are directly imported. Do not include any code
    other than declaration of the necessary dictionaries.
"""

# Try Fs("book@en").items() to get names of the book included
# in the text-fabric dataset
ot_train_books = {
    "Genesis": list(range(1, 50 + 1)),
    "Exodus": list(range(1, 21 + 1)),
}

ot_test_books = {"Deuteronomy": list(range(1, 20 + 1))}

ot_prod_books = {
    "Joshua": list(range(1, 24 + 1)),
    "Judges": list(range(1, 21 + 1)),
    "Samuel_1": list(range(1, 32 + 1)),
    "Samuel_2": list(range(1, 24 + 1)),
    "Kings_1": list(range(1, 22 + 1)),
    "Kings_2": list(range(1, 26 + 1)),
    "Ezra": list(range(1, 10 + 1)),
    "Nehemia": list(range(1, 13 + 1)),
    "Chronicles_1": list(range(1, 29 + 1)),
    "Chronicles_2": list(range(1, 36 + 1)),
    "Maccabees_1_A": list(range(1, 16 + 1)),
    "Maccabees_1_B": list(range(1, 16 + 1)),
    "Ruth": list(range(1, 4 + 1)),
    "Esther": list(range(1, 10 + 1)),
}

nt_train_books = {
    "Matthew": list(range(1, 30 + 1)),
    "Mark": list(range(1, 16 + 1)),
    "Luke": list(range(1, 24 + 1)),
    "John": list(range(1, 21 + 1)),
}

nt_test_books = {"Acts": list(range(1, 28 + 1))}
