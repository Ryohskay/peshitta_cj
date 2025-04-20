from flask import Flask, render_template, g, request, jsonify
from flask import request
from werkzeug.local import LocalProxy

from src.classifier.result_utils import Verse
from src.classifier.fname_utils import SavefileName

from src.ui_web.load_predictions import load_preds
from src.ui_web.load_book_probas import BookProbas
from src.ui_web.select_load_classifier import ClassifierConfig, ResultFilesIndex

from typing import Any


app = Flask(__name__)

v = Verse("Chronicles_1",
        "1 Chronicles Chapter 01 Verse 33",
        ['WB"NJ', "MDJN", "<P>", "W><PR", "WXNWK", "W>BJD<", "W>LR<>", "HLJN", "KLHWN", 'BN"JH^', "DQNVWR>"],
        syriac_words=["ܘܒ̈ܢܝ", "ܡܕܝܢ", "ܥܦܐ", "ܘܐܥܦܪ", "ܘܚܢܘܟ", "ܘܐܒܝܕܥ", "ܘܐܠܪܥܐ", "ܗܠܝܢ", "ܟܠܗܘܢ", "ܒܢ̈ܝܗ̇", "ܕܩܢܛܘܪܐ"],
        origin="ETCBC"
        )

file_index = ResultFilesIndex()

def parse_clf_configs(req: LocalProxy) -> ClassifierConfig:
    """Parse the request arguments into ``ClassifierConfig``.

    Returns:
        an instance of
        :class:`src.ui_web.select_load_classifier.ClassifierConfig`.
    """
    result = ClassifierConfig()
    for field in ClassifierConfig.fields():
        # loop over the predefined configurations to avoid
        # picking up unwanted values
        config_val = req.form.get(field.name)
        if config_val is not None:
            # cast the value to the correct type
            if field.type is bool:
                setattr(result, field.name, (config_val.upper() == "TRUE"))
            elif field.type is int:
                setattr(result, field.name, int(config_val))
            else:
                # cast to str by default
                setattr(result, field.name, str(config_val))

@app.route("/")
def display_default():
    # view: verses of a particular classifier's results
    # show results of a plain ETCBC classifier by default
    # > display each book on one page,
    # > where each page has blocks, each with the verse in:
    # > syriac, etcbc, cal scripts and then probas predicted
    # > also display book level production results
    # >> (book probas at .9, .8, .5 threshold?)
    return render_template("verses.html", verses=[v])

# TODO (essential): accept a GET request with some classifier configurations
# and return the results of that classifier

# Code adapted from https://github.com/pallets/flask/blob/main/examples/javascript/js_example
# (Accessed: 15 April 2025)
@app.route("/classifier")
def find_classifier():
    clf_configs = parse_clf_configs(request)
    save_file = file_index.match_files_by_config(clf_configs)
    return jsonify()
