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
from flask import Flask, abort, render_template, request
from werkzeug.local import LocalProxy

from src.classifier.fname_utils import FnameExtraOpts
from src.classifier.result_utils import Verse
from src.shared.label_data import ValToLabel
from src.ui_web.select_load_classifier import (
    ClassifierConfig,
    ClassifierResultsModel,
    ResultFilesIndex,
)

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
        # picking up unexpected parameters
        if field.name == "extra_opts":
            # extra_opts is a list of options
            extra_opts = req.args.getlist(field.name)
            if extra_opts is not None:
                # cast the value to the correct type
                setattr(
                    result,
                    field.name,
                    [FnameExtraOpts[opt.split(".")[-1]] for opt in extra_opts],
                )
        else:
            # get other values as single value params
            config_val = req.args.get(field.name)
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


@app.route("/get_verses")
def get_verses():
    """Render the classifier results based on the request parameters."""
    conf = parse_clf_configs(request)
    cr = ClassifierResultsModel(conf, file_index, "./src/classifier/out")
    cr.load_results()
    print(cr.get_book_verses())
    return render_template("verses.html", results=cr, label_map=ValToLabel)


@app.route("/")
def display_default():
    """Render the default page."""
    # view: verses of a particular classifier's results
    # show results of a plain ETCBC classifier by default
    # > display each book on one page,
    # > where each page has blocks, each with the verse in:
    # > syriac, etcbc, cal scripts and then probas predicted
    # > also display book level production results
    # >> (book probas at .9, .8, .5 threshold?)
    conf = ClassifierConfig(
        name="mnb",
        origin="ETCBC",
        is_n_gram=True,
        n=3,
        is_bow=True,
        is_char_level=True,
        extra_opts=[
            FnameExtraOpts.REMOVE_DIACRITICS,
            FnameExtraOpts.REMOVE_PROPN,
            FnameExtraOpts.REMOVE_FROM_BOTH,
        ],
    )
    return get_verses(conf)


# Code adapted from https://github.com/pallets/flask/blob/main/examples/javascript/js_example
# (Accessed: 15 April 2025)
@app.route("/get_classifier")
def get_classifier():
    """Render the classifier results based on the request parameters."""
    clf_configs = parse_clf_configs(request)
    print(clf_configs)
    files = file_index.match_files_by_config(clf_configs)
    for file in files:
        if file.is_clf_summary:
            cr = ClassifierResultsModel(
                clf_configs, file_index, "./src/classifier/out"
            )
            cr.load_results()
            return render_template("clf_summary.html", results=cr)
    # if no file is found, return 404
    abort(404)  # Not found
    return None
    # code after abort is never executed
