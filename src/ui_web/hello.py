from flask import Flask, render_template
from src.classifier.result_utils import Verse

import src.ui_web.load_predictions
import src.ui_web.load_book_probas

app = Flask(__name__)

v = Verse("Chronicles_1",
        "1 Chronicles Chapter 01 Verse 33",
        ['WB"NJ', "MDJN", "<P>", "W><PR", "WXNWK", "W>BJD<", "W>LR<>", "HLJN", "KLHWN", 'BN"JH^', "DQNVWR>"],
        syriac_words=["ܘܒ̈ܢܝ", "ܡܕܝܢ", "ܥܦܐ", "ܘܐܥܦܪ", "ܘܚܢܘܟ", "ܘܐܒܝܕܥ", "ܘܐܠܪܥܐ", "ܗܠܝܢ", "ܟܠܗܘܢ", "ܒܢ̈ܝܗ̇", "ܕܩܢܛܘܪܐ"],
        origin="ETCBC"
        )

# func: construct a file name based on options

# func: open a file and load the results there
# > receive a file name and open the file with that name

@app.route("/")
def get_verses():
        # view: verses of a particular classifier's results
        # > display each book on one page,
        # > where each page has blocks, each with the verse in:
        # > syriac, etcbc, cal scripts and then probas predicted
        # > also display book level production results
        # >> (book probas at .9, .8, .5 threshold?)
        return render_template("verses.html", verses=[v])
