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

from classifier.cal_aa_eval import csvify_cal, remove_proper_nouns
from classifier.eval_utils import eval_and_save
from classifier.load_cal import load_cal_dataset
from classifier.prediction_utils import predict_proba
from classifier.wrappers import BoWEstimator

if __name__ == "__main__":
    cal_loaded = load_cal_dataset("./")

    train_x = cal_loaded.train.get_samples()
    train_y = cal_loaded.train.get_labels()

    print("CAL --->")
    print("MultinomialNB")
    print("> Plain Classifier")
    print(">> Training")
    c_mnb = BoWEstimator(MultinomialNB(), " ".join)
    c_mnb.fit(train_x, train_y)
    print(">> Quick Evaluation")
    eval_and_save(c_mnb, cal_loaded, csvify_cal, out_dir="./out/",
                  save_file_prefix="just_to_check_")

    print("Production Data")
    ot_prod = cal_loaded.production

    preds = predict_proba(c_mnb, cal_loaded.production)

    save_file = Path("PRODUCTION_mnb_cal_prediction_proba_all.csv")
    preds.save_to_file(csvify_cal, save_file)

    print("> Remove PN, GN & underscores from the training set")
    print(">> Training")
    # Remove personal names and place names from the training data
    # and train new classifiers
    train_x_removed = remove_proper_nouns(train_x)

    c_mnb_r = BoWEstimator(MultinomialNB(), " ".join)
    c_mnb_r.fit(train_x_removed, train_y)

    print(">> Quick Evaluation")
    eval_and_save(c_mnb_r, cal_loaded, csvify_cal, out_dir="./out/",
                  save_file_prefix="just_to_check_")

    print("Production Data")
    ot_prod = cal_loaded.production

    preds = predict_proba(c_mnb, cal_loaded.production)

    save_file = Path("PRODUCTION_mnb_cal_prediction_proba_all.csv")
    preds.save_to_file(csvify_cal, save_file)



    print("CAL --->")
    print("MultinomialNB")
    print("> Plain Classifier")
    print(">> Training")
    c_mnb = BoWEstimator(MultinomialNB(), " ".join)
    c_mnb.fit(train_x, train_y)
    print(">> Quick Evaluation")
    eval_and_save(c_mnb, cal_loaded, csvify_cal, out_dir="./out/",
                  save_file_prefix="just_to_check_")

    print("Production Data")
    ot_prod = cal_loaded.production

    preds = predict_proba(c_mnb, cal_loaded.production)

    save_file = Path("PRODUCTION_mnb_cal_prediction_proba_all.csv")
    preds.save_to_file(csvify_cal, save_file)

