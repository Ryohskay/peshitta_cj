
from pathlib import Path

from sklearn.naive_bayes import MultinomialNB

import classifier.book_data as book_data
from classifier.cal_aa_eval import csvify_cal, remove_proper_nouns
from classifier.eval_utils import eval_and_save, convert
from classifier.load_cal import get_book_verses, load_df_json
from classifier.result_utils import ProbaPredictions
from classifier.wrappers import BoWEstimator

if __name__ == "__main__":
    PROJ_ROOT = Path("./")
    print(f"Loading CAL data from {PROJ_ROOT}")
    df = load_df_json(PROJ_ROOT / "scraper/cal_results/")

    if df is None:
        msg = f"Failed to fetch data from {PROJ_ROOT / 'scraper/cal_results/'}"
        raise RuntimeError(msg)

    # get the training data
    print("OT_train")
    ot_train_verses = get_book_verses(
        df, book_data.ot_train_books, trim_none=True
    )
    print("NT_train")
    nt_train_verses = get_book_verses(
        df, book_data.nt_train_books, trim_none=True
    )

    train_x = ot_train_verses.copy()
    train_x.extend(nt_train_verses)
    train_y = [0 for v in ot_train_verses]
    train_y.extend([1 for v in nt_train_verses])

    # get the test data
    ot_test_verses = get_book_verses(
        df, book_data.ot_test_books, trim_none=True
    )
    nt_test_verses = get_book_verses(
        df, book_data.nt_test_books, trim_none=True
    )

    # get the production data
    ot_prod = get_book_verses(df, book_data.ot_prod_books, trim_none=True)

    print("\nProduction Data")
    print("\nPlain Classifier")
    print("MultinomialNB")

    c_mnb = BoWEstimator(MultinomialNB(), " ".join)
    c_mnb.fit(train_x, train_y)

    print("OT --->")
    y_pred_proba = c_mnb.predict_proba(ot_prod)
    # convert the list of probas to a list of labels
    y_pred = convert(y_pred_proba, thresh=0.5)

    proba_preds = ProbaPredictions(ot_prod, y_pred,
                     y_pred_proba)

    y_all = proba_preds.predictions
    probas = proba_preds.get_probas()

    c_mnb, probas_pair_r = eval_and_save(
                                c_mnb,
                                ot_test_verses,
                                nt_test_verses,
                                csvify_cal,
                                out_dir="./classifier/out/",
                                save_file_prefix="PRODUCTION_cal_",
                                save_file_suffix="_plain"
                            )
    save_file = Path(
        "PRODUCTION_prediction_all" + kw ".csv"
    )

    print("> Remove PN & GN from the training set")
    print("MultinomialNB")
    # Remove personal names and place names from the training data
    # and train new classifiers
    train_x_removed = remove_proper_nouns(train_x)

    c_mnb_r = BoWEstimator(MultinomialNB(), " ".join)
    c_mnb_r.fit(train_x_removed, train_y)

    c_mnb_r, probas_pair_r = eval_and_save(
                                c_mnb_r,
                                ot_test_verses,
                                nt_test_verses,
                                csvify_cal,
                                out_dir="./classifier/out/",
                                save_file_prefix="PRODUCTION_cal_",
                                save_file_suffix="_removed_both"
                            )
