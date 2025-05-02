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
"""Load the classifier statistics from the summary files"""
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TypedDict

from src.classifier.fname_utils import SavefileName
from src.classifier.result_utils import (
    ResultStatsDict,
)


class JsonifiedSummaryDict(TypedDict):
    """TypedDict for the JSON summary file."""

    n_gram_form: Literal["word", "char"]  # the n-gram form (word or char).
    n: int  # the n for n-grams.
    total_n_grams_parsed: int  # the total number of n-grams parsed.
    top_ten_in_training: list[tuple[Iterable[str], int]]  # the top ten
    # the percentages of the mislabelled verses in the test set
    test_mislabel_percent: dict[str, float]
    # dictionary containing the metrics, of
    # :class:``src.classifier.result_utils.ResultStatsDict`` type.
    metrics: ResultStatsDict


def load_clf_stats(
    json_fname: SavefileName, load_dir: str | Path = "src/classifier/out/"
) -> JsonifiedSummaryDict:
    """Load the classifier statistics from the "summary" files.

    Returns:
        dict with scheme :class:`src.ui_web.load_clf_stats.JsonifiedSummaryDict`
        containing the classifier statistics.
    """
    # check if the file is a summary file
    if not json_fname.is_clf_summary:
        msg = f"File {json_fname} is not a classifier evaluation summary file."
        raise ValueError(msg)
    # load the classifier statistics
    fname_p = Path(load_dir) / json_fname.get_fname()
    with fname_p.open(encoding="utf-8") as fp:
        data: JsonifiedSummaryDict = json.load(fp)

    return data
