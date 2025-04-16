from flask import Flask, render_template
from src.classifier.result_utils import Verse

app = Flask(__name__)

v = Verse("Chronicles_1",
        "1 Chronicles Chapter 01 Verse 33",
        ['WB"NJ', "MDJN", "<P>", "W><PR", "WXNWK", "W>BJD<", "W>LR<>", "HLJN", "KLHWN", 'BN"JH^', "DQNVWR>"],
        syriac_words=["ܘܒ̈ܢܝ", "ܡܕܝܢ", "ܥܦܐ", "ܘܐܥܦܪ", "ܘܚܢܘܟ", "ܘܐܒܝܕܥ", "ܘܐܠܪܥܐ", "ܗܠܝܢ", "ܟܠܗܘܢ", "ܒܢ̈ܝܗ̇", "ܕܩܢܛܘܪܐ"],
        origin="ETCBC"
        )

@app.route("/")
def get_verses():
    return render_template("verses.html", verses=[v])
