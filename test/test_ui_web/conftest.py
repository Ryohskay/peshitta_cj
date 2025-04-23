import pytest
from src.classifier.fname_utils import FnameExtraOpts, SavefileName


@pytest.fixture
def mock_save_fname():
    """Fixture to provide a mock SavefileName object."""
    fname = SavefileName(origin="CAL", classifier_alias="mnb", file_ext="csv")
    fname.set_ngram_opts(n=3, is_n_gram=True, is_bow=True, is_char_level=True)
    fname.add_extra_opts(
        [
            FnameExtraOpts.REMOVE_PROPN,
            FnameExtraOpts.REMOVE_UNDERSCORES,
            FnameExtraOpts.REMOVE_FROM_BOTH,
        ]
    )
    fname.mark_special_file(is_total_proba=True)
    return fname
