import json
from pathlib import Path
from typing import Literal, TypedDict

from src.classifier.fname_utils import SavefileName
from src.classifier.result_utils import (
    ResultStatsDict,
)


class JsonifiedSummaryDict(TypedDict):
    """TypedDict for the JSON summary file.

    Attributes:
        n_gram_form: the n-gram form (word or char).
        n: the n for n-grams.
        top_ten_in_training: the top ten n-grams in the training set.
        test_mislabel_percent: the mislabel percentages for the test set.
        total_n_grams_parsed: the total number of n-grams parsed.
        metrics: dictionary containing the metrics, of
            :class:``src.classifier.result_utils.ResultStatsDict`` type.
    """

    n_gram_form: Literal["word", "char"]
    n: int
    total_n_grams_parsed: int
    top_ten_in_training: list[tuple[tuple[str], int]]
    test_mislabel_percent: dict[str, float]
    metrics: ResultStatsDict


def load_clf_stats(
    json_fname: SavefileName, load_dir: str | Path = "src/classifier/out/"
) -> JsonifiedSummaryDict:
    """Load the classifier statistics from the "summary" files.

    Returns:
        A list of :class:`src.classifier.result_utils.ResultStats` instances
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
