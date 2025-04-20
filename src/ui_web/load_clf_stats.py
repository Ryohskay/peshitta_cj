from src.classifier.fname_utils import SavefileName
from src.classifier.result_utils import ResultStats, ThresholdStats
from pathlib import Path
import json

def load_clf_stats(json_fname: SavefileName) -> ResultStats:
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
    fname_p = Path(json_fname.get_fname())
    with fname_p.open(encoding="utf-8") as fp:
        data = json.load(fp)

    clf_stats = data["metrics"]
