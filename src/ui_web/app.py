from flask import Flask, jsonify, render_template, request
from werkzeug.local import LocalProxy

from src.classifier.result_utils import Verse
from src.ui_web.select_load_classifier import ClassifierConfig, ResultFilesIndex

app = Flask(__name__)

v = Verse(
    "Chronicles_1",
    "1 Chronicles Chapter 01 Verse 33",
    [
        'WB"NJ',
        "MDJN",
        "<P>",
        "W><PR",
        "WXNWK",
        "W>BJD<",
        "W>LR<>",
        "HLJN",
        "KLHWN",
        'BN"JH^',
        "DQNVWR>",
    ],
    syriac_words=[
        "ܘܒ̈ܢܝ",
        "ܡܕܝܢ",
        "ܥܦܐ",
        "ܘܐܥܦܪ",
        "ܘܚܢܘܟ",
        "ܘܐܒܝܕܥ",
        "ܘܐܠܪܥܐ",
        "ܗܠܝܢ",
        "ܟܠܗܘܢ",
        "ܒܢ̈ܝܗ̇",
        "ܕܩܢܛܘܪܐ",
    ],
    origin="ETCBC",
)

file_index = ResultFilesIndex()


def parse_clf_configs(req: LocalProxy) -> ClassifierConfig:
    """Parse the request arguments into ``ClassifierConfig``.

    Returns:
        an instance of
        :class:`src.ui_web.select_load_classifier.ClassifierConfig`.
    """
    result = ClassifierConfig()
    for field in result.fields():
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
                # pass the value as is by default
                setattr(result, field.name, config_val)
    return result


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
@app.route("/get_classifier")
def get_classifier_summary():
    clf_configs = parse_clf_configs(request)
    files = file_index.match_files_by_config(clf_configs)
    for file in files:
        if file.is_clf_summary:
            return jsonify(file.get_summary())
    flask.abort(404)  # Not found
    # code after abort is never executed
