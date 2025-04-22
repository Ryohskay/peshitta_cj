import numpy as np
import pytest

from src.classifier.dataset_skeleton import LoadedDataset
from src.classifier.prediction_utils import convert, predict, predict_proba
from src.classifier.result_utils import Predictions, ProbaPredictions
from src.classifier.wrappers import Classifier, ProbaClassifier


class TestPredict:
    def test_predict(
        self, mnb_classifier: Classifier, loaded_etcbc: LoadedDataset
    ) -> None:
        mnb_classifier.fit(
            loaded_etcbc.train.get_samples(), loaded_etcbc.train.get_labels()
        )
        predictions = predict(
            mnb_classifier,
            loaded_etcbc.test.get_samples(),
            loaded_etcbc.test.get_labels(),
        )
        assert isinstance(predictions, Predictions)
        assert len(predictions.samples) == len(loaded_etcbc.test.get_samples())
        assert len(predictions.predictions) == len(
            loaded_etcbc.test.get_samples()
        )
        assert predictions.correct_labels is not None
        assert len(predictions.correct_labels) == len(
            loaded_etcbc.test.get_labels()
        )


class TestConvert:
    def test_convert(self) -> None:
        # with threshold 0.5
        probas = [[0.8, 0.2], [0.3, 0.7], [0.1, 0.9], [0.5, 0.5]]
        result = convert(probas, 0.5)
        expected = np.array([0, 1, 1, -1])
        assert np.array_equal(result, expected)
        # with threshold higher than 0.5
        result = convert(probas, 0.75)
        expected = np.array([0, -1, 1, -1])
        assert np.array_equal(result, expected)
        # with threshold lower than 0.5
        result = convert(probas, 0.2)
        expected = np.array([0, -1, 1, -1])
        assert np.array_equal(result, expected)

    def test_convert_raises(self) -> None:
        probas = [[0.1, 0.2], [0.3, 0.0]]
        threshold = 0.5
        with pytest.raises(
            ValueError,
            match="Something is wrong with the probability at index: 0!",
        ):
            convert(probas, threshold)


class TestPredictProba:
    def test_predict_proba(
        self, mnb_classifier: ProbaClassifier, loaded_etcbc: LoadedDataset
    ) -> None:
        mnb_classifier.fit(
            loaded_etcbc.train.get_samples(), loaded_etcbc.train.get_labels()
        )
        proba_predictions = predict_proba(
            mnb_classifier,
            loaded_etcbc.test.get_samples(),
            loaded_etcbc.test.get_labels(),
            threshold=0.5,
        )
        assert isinstance(proba_predictions, ProbaPredictions)
        assert len(proba_predictions.samples) == len(
            loaded_etcbc.test.get_samples()
        )
        assert len(proba_predictions.predictions) == len(
            loaded_etcbc.test.get_samples()
        )
        assert len(proba_predictions._probas) == len(
            loaded_etcbc.test.get_samples()
        )
        assert proba_predictions.correct_labels is not None
        assert len(proba_predictions.correct_labels) == len(
            loaded_etcbc.test.get_labels()
        )
