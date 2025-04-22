import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.classifier.production import predict_on_prod
from src.classifier.wrappers import BoWEstimator
from src.classifier.fname_utils import SavefileName
from src.classifier.dataset_skeleton import LoadedDataset

def test_predict_on_prod(
    monkeypatch,
    mock_eval_and_save,
    mock_predict_proba,
    mnb_classifier,
    loaded_etcbc,
    etcbc_base_savefile: SavefileName,
    tmp_path: Path
):
    """Test the predict_on_prod function."""
    # Mock the behaviour of predict_proba
    monkeypatch.setattr("src.classifier.prediction_utils.predict_proba", mock_predict_proba)
    monkeypatch.setattr("src.classifier.production.eval_and_save", mock_eval_and_save)

    # Call the function
    predict_on_prod(
        mnb_classifier,
        loaded_etcbc,
        etcbc_base_savefile,
        save_dir=str(tmp_path),
        thresh=0.5,
    )

    # Check that correct methods/functions were called
    mock_eval_and_save.assert_called_once()

    # Check if files are saved correctly
    etcbc_base_savefile.mark_special_file(is_prod=True)
    assert (tmp_path / etcbc_base_savefile.get_fname()).exists()
    assert (tmp_path / etcbc_base_savefile.get_fname()).is_file()
