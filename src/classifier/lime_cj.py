"""Integrate Lime Explainer into this project."""

# Code adapted from: https://marcotcr.github.io/lime/tutorials/Lime%20-%20basic%20usage%2C%20two%20class%20case.html
# (Accessed: 15 April 2025)

from lime.lime_text import LimeTextExplainer
from sklearn.naive_bayes import MultinomialNB
from src.classifier.result_utils import Verse

from src.classifier.eval_utils import (
    eval_and_save,
)
from src.classifier.textfabric_utils import load_etcbc_dataset
from src.classifier.wrappers import BoWEstimator

from src.classifier.etcbc_aa_eval import remove_non_chars, csvify_etcbc


def splitter(v: Verse):
    return v.get_syriac_words()


loaded_etc = load_etcbc_dataset()
train_verses = remove_non_chars(loaded_etc.train.get_samples())
train_verse_labels = loaded_etc.train.get_labels()
ot_test_verses = remove_non_chars(loaded_etc.test.get_samples(0))
nt_test_verses = remove_non_chars(loaded_etc.test.get_samples(1))


print("\nPlain Classifier")
mnb = BoWEstimator(MultinomialNB(), " ".join)
mnb.fit(train_verses, train_verse_labels)

# evaluate and save results
eval_and_save(
        mnb,
        loaded_etc,
        csvify_etcbc,
        out_dir="./src/classifier/out/",
        save_file_prefix="etcbc_"
        )

class_names = ["Jewish", "Christian"]
explainer = LimeTextExplainer(class_names=class_names, char_level=True)

idx = 83
exp = explainer.explain_instance(
        " ".join(loaded_etc.test.get_samples()[idx].get_translit_words()),
                     mnb.predict_proba_translit, num_features=10)
print("Document id: %d" % idx)
print("Probability(christian) =", mnb.predict_proba([loaded_etc.test.get_samples()[idx]])[1])
print("True class: %s" % class_names[loaded_etc.test.get_labels()[idx]])
