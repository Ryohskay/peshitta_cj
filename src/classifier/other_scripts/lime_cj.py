"""Integrate Lime Explainer into this project."""

# Code adapted from: https://marcotcr.github.io/lime/tutorials/Lime%20-%20basic%20usage%2C%20two%20class%20case.html
# (Accessed: 15 April 2025)

from lime.lime_text import LimeTextExplainer
from sklearn.naive_bayes import MultinomialNB

from src.classifier.etcbc_aa_eval import (
    etcbc_eval_classifier,
    remove_non_chars,
)
from src.classifier.fname_utils import FnameExtraOpts, SavefileName
from src.classifier.textfabric_utils import load_etcbc_dataset
from src.classifier.wrappers import BoWEstimator
from src.shared.label_data import ValToLabel

loaded_etc = load_etcbc_dataset()

n_window = 3
print("\nPlain Classifier")
mnb = BoWEstimator(MultinomialNB(), " ".join, n=n_window)
mnb.set_preprocessor(remove_non_chars)

# evaluate and save results
save_fname = SavefileName("ETCBC", "mnb")
save_fname.set_ngram_opts(n=n_window)
save_fname.add_extra_opts([FnameExtraOpts.REMOVE_DIACRITICS])
etcbc_eval_classifier(mnb, loaded_etc, save_fname)

explainer = LimeTextExplainer(
    class_names=list[ValToLabel.values()], char_level=True
)

idx = 83
exp = explainer.explain_instance(
    " ".join(loaded_etc.test.get_samples()[idx].get_translit_words()),
    mnb.predict_proba,
    num_features=4,
)
print("Document id: %d" % idx)
print(
    "Probability(christian) =",
    mnb.predict_proba([loaded_etc.test.get_samples()[idx]])[0][1],
)
print("True class: %s" % ValToLabel[loaded_etc.test.get_labels()[idx]])

exp.save_to_file("./out/lime_explainer.html", text=True)
