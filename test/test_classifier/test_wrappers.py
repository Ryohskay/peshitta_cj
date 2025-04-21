
import pytest
from sklearn.naive_bayes import MultinomialNB

from src.classifier.dataset_skeleton import LoadedDataset
from src.classifier.result_utils import Verse
from src.classifier.wrappers import BoWEstimator, Classifier, ProbaClassifier


class TestClassifier:
    def test_init_valid(self):
        clf = Classifier(MultinomialNB())
        assert hasattr(clf.algo, "fit")
        assert hasattr(clf.algo, "predict")

    def test_init_invalid(self):
        with pytest.raises(
            ValueError, match="clf .+ does not have a method named `.fit`"
        ):
            Classifier(
                object()  # type: ignore[reportArgumentType]
            )  # an empty object without `.fit` method

    def test_fit_and_predict(
        self, mnb_classifier: BoWEstimator, loaded_etcbc: LoadedDataset
    ):
        mnb_classifier.fit(
            loaded_etcbc.train.get_samples(), loaded_etcbc.train.get_labels()
        )
        predictions = mnb_classifier.predict(loaded_etcbc.test.get_samples())
        assert len(predictions) == len(loaded_etcbc.test.get_samples())


class TestProbaClassifier:
    def test_init_valid(self):
        clf = ProbaClassifier(MultinomialNB())
        assert hasattr(clf.algo, "predict_proba")

    def test_init_invalid(self):
        with pytest.raises(
            ValueError,
            match="clf .* does not have a method named `predict_proba`",
        ):
            ProbaClassifier(
                object()  # type: ignore[reportArgumentType]
            )  # an empty object without `.predict_proba` method

    def test_predict_proba(
        self, mnb_classifier: BoWEstimator, loaded_etcbc: LoadedDataset
    ):
        mnb_classifier.fit(
            loaded_etcbc.train.get_samples(), loaded_etcbc.train.get_labels()
        )
        probas = mnb_classifier.predict_proba(loaded_etcbc.test.get_samples())
        assert len(probas) == len(loaded_etcbc.test.get_samples())
        assert len(probas[0]) == 2  # Binary classification


class TestBoWEstimator:
    def test_init(self):
        clf = BoWEstimator(MultinomialNB(), formatter=lambda x: x, n=2)
        assert clf.n == 2
        assert callable(clf.n_gram_formatter)

    def mock_preprocessor(self, words: list[str]) -> list[str]:
        return [word.upper() for word in words]

    def test_set_preprocessor(self, mnb_classifier: BoWEstimator):
        mnb_classifier.set_preprocessor(self.mock_preprocessor)
        assert mnb_classifier.preprocessor == self.mock_preprocessor

    def test_process_verses(
        self, mnb_classifier: BoWEstimator, cal_verse: Verse
    ):
        # test the code where there is no preprocessor defined
        processed = mnb_classifier._process_verses(
            [cal_verse.get_translit_words()]
        )
        assert processed[0] == cal_verse.get_translit_words()
        # test the code when the preprocessor removes a particular str
        mnb_classifier.set_preprocessor(self.mock_preprocessor)
        processed = mnb_classifier._process_verses(
            [cal_verse.get_translit_words()]
        )
        assert processed is not None
        assert processed[0] == [
            w.upper() for w in cal_verse.get_translit_words()
        ]

    def test_fit(
        self, mnb_classifier: BoWEstimator, loaded_etcbc: LoadedDataset
    ):
        mnb_classifier.fit(
            loaded_etcbc.train.get_samples(), loaded_etcbc.train.get_labels()
        )
        assert mnb_classifier.vocabs is not None
        assert len(mnb_classifier.train_vector) == len(
            loaded_etcbc.train.get_samples()
        )

    def test_fit_raises(self, mnb_classifier: BoWEstimator, cal_verse: Verse):
        with pytest.raises(
            ValueError, match=r"Length of X \(\d+\) and y \(\d+\) do not match."
        ):
            mnb_classifier.fit([cal_verse], [0, 1])

    def test_predict(
        self, mnb_classifier: BoWEstimator, loaded_etcbc: LoadedDataset
    ):
        mnb_classifier.fit(
            loaded_etcbc.train.get_samples(), loaded_etcbc.train.get_labels()
        )
        predictions = mnb_classifier.predict(loaded_etcbc.test.get_samples())
        assert len(predictions) == len(loaded_etcbc.test.get_samples())

    def test_predict_raises(self, mnb_classifier: BoWEstimator):
        with pytest.raises(
            ValueError, match="Cannot fetch the vocabulary of the model."
        ):
            # .predict called before the model is fit on data
            mnb_classifier.predict([["word1", "word2"]])

    def test_predict_proba(
        self, mnb_classifier: BoWEstimator, loaded_etcbc: LoadedDataset
    ):
        mnb_classifier.fit(
            loaded_etcbc.train.get_samples(), loaded_etcbc.train.get_labels()
        )
        probabilities = mnb_classifier.predict_proba(
            loaded_etcbc.test.get_samples()
        )
        assert len(probabilities) == len(loaded_etcbc.test.get_samples())
        # make sure it's represented as a binary classification
        assert len(probabilities[0]) == 2

    def test_predict_proba_raises(self, mnb_classifier: BoWEstimator):
        with pytest.raises(
            ValueError, match="Cannot fetch the vocabulary of the model."
        ):
            # .predict called before the model is fit on data
            mnb_classifier.predict_proba([["word1", "word2"]])
