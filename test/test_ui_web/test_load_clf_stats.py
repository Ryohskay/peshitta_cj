from unittest import mock
import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock
from src.ui_web.load_clf_stats import load_clf_stats
from src.classifier.fname_utils import SavefileName
from src.classifier.result_utils import ResultStats


@pytest.fixture
def mock_savefile_name() -> SavefileName:
    """Fixture to provide a mock SavefileName object."""
    savefile = SavefileName(origin="CAL", classifier_alias="mnb", file_ext="json")
    savefile.mark_special_file(is_clf_summary=True)
    savefile.set_ngram_opts(
        n=3,
        is_n_gram=True,
        is_bow=True,
        is_char_level=True,
    )
    return savefile

def test_load_clf_stats(mock_savefile_name: SavefileName):
    """Test loading a valid classifier statistics file."""
    asset_dir = "assets/classifier_results"

    # Call the function
    assert mock_savefile_name.is_clf_summary
    result = load_clf_stats(mock_savefile_name, load_dir=asset_dir)

    # Assertions
    assert isinstance(result, dict)
    assert result["metrics"]["supports"] == [557, 1007]
    assert len(result["metrics"]["thresh_stats"]) == 4
    assert result["metrics"]["thresh_stats"][0]["threshold"] == 0.5
    assert result["metrics"]["thresh_stats"][1]["threshold"] == 0.8
    assert result["metrics"]["thresh_stats"][2]["threshold"] == 0.9
    assert result["metrics"]["thresh_stats"][3]["threshold"] == 0.95


def test_load_clf_stats_raises(
        mock_savefile_name: SavefileName,
        tmp_path: Path,
        ):
    """Test loading invalid files."""
    with pytest.raises(FileNotFoundError):
        load_clf_stats(mock_savefile_name, load_dir=tmp_path)

    mock_savefile_name.is_clf_summary = False
    asset_dir = "assets/classifier_results"

    with pytest.raises(ValueError, match="is not a classifier evaluation summary file"):
        load_clf_stats(mock_savefile_name, load_dir=asset_dir)
