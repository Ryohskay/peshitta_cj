# BSD 2-Clause License
#
# Copyright (c) 2025, Ryosuke Nagata
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
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

from pathlib import Path

from sklearn.naive_bayes import MultinomialNB

from src.classifier.cal_aa_eval import (
    csvify_cal,
    remove_proper_nouns,
    remove_underscores,
)
from src.classifier.etcbc_aa_eval import csvify_etcbc
from src.classifier.eval_utils import eval_and_save
from src.classifier.load_cal import load_cal_dataset
from src.classifier.prediction_utils import predict_proba
from src.classifier.textfabric_utils import load_etcbc_dataset
from src.classifier.wrappers import BoWEstimator

if __name__ == "__main__":
    # LOAD ETCBC DATA
    etcbc_loaded = load_etcbc_dataset()
    train_x_etc = etcbc_loaded.train.get_samples()
    train_y_etc = etcbc_loaded.train.get_labels()
    print("ETCBC --->")
    print("MultinomialNB")
    print("> Plain Classifier")
    print(">> Training")
    c_mnb = BoWEstimator(MultinomialNB(), " ".join)
    c_mnb.fit(train_x_etc, train_y_etc)
    print(">> Quick Evaluation")
    eval_and_save(c_mnb, etcbc_loaded, csvify_cal, out_dir="./classifier/out/",
                  save_file_prefix="just_to_check_")

    print("Production Data")
    ot_prod = etcbc_loaded.production

    preds = predict_proba(c_mnb, etcbc_loaded.production)

    save_file = Path("./classifier/out/PRODUCTION_mnb_etcbca_"
                     + "prediction_proba_all.csv")
    preds.save_to_file(csvify_etcbc, save_file)

    print("> Character unigram classifier")
    print(">> Training")
    c_mnb = BoWEstimator(MultinomialNB(), " ".join, n=1)
    c_mnb.fit(train_x_etc, train_y_etc)
    print(">> Quick Evaluation")
    eval_and_save(c_mnb, etcbc_loaded, csvify_cal, out_dir="./classifier/out/",
                  save_file_prefix="just_to_check_")

    print("Production Data")
    ot_prod = etcbc_loaded.production

    preds = predict_proba(c_mnb, etcbc_loaded.production)

    save_file = Path("./classifier/out/PRODUCTION_mnb_etcbc_"
                     + "prediction_proba_all_char_unigram.csv")
    preds.save_to_file(csvify_etcbc, save_file)

    print("CAL --->")
    # LOAD CAL DATA
    cal_loaded = load_cal_dataset("./")

    train_x_cal = cal_loaded.train.get_samples()
    train_y_cal = cal_loaded.train.get_labels()
    print("MultinomialNB")
    print("> Plain Classifier")
    print(">> Training")
    c_mnb = BoWEstimator(MultinomialNB(), " ".join)
    c_mnb.fit(train_x_cal, train_y_cal)
    print(">> Quick Evaluation")
    eval_and_save(c_mnb, cal_loaded, csvify_cal, out_dir="./classifier/out/",
                  save_file_prefix="just_to_check_")

    print("Production Data")
    ot_prod = cal_loaded.production

    preds = predict_proba(c_mnb, cal_loaded.production)

    save_file = Path("./classifier/out/PRODUCTION_mnb_cal_"
                     + "prediction_proba_all.csv")
    preds.save_to_file(csvify_cal, save_file)

    print("> Remove PN, GN & underscores from the training & test set")
    print(">> Training")
    # Remove personal names and place names from the training data
    # and train new classifiers
    train_x_removed = remove_proper_nouns(remove_underscores(train_x_cal))

    c_mnb_r = BoWEstimator(MultinomialNB(), " ".join)
    c_mnb_r.fit(train_x_removed, train_y_cal)

    print(">> Quick Evaluation")
    eval_and_save(c_mnb_r, cal_loaded, csvify_cal, out_dir="./classifier/out/",
                  save_file_prefix="just_to_check_")

    print("Production Data")
    ot_prod = remove_proper_nouns(remove_underscores(cal_loaded.production))

    preds = predict_proba(c_mnb, ot_prod)

    save_file = Path("./classifier/out/PRODUCTION_mnb_cal_"
                     + "prediction_proba_all_both_removed.csv")
    preds.save_to_file(csvify_cal, save_file)
