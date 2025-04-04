from sklearn.naive_bayes import MultinomialNB
from classifier.eval_utils import evaluate_classifier, save_all_preds, save_mislabels
from classifier.fitting_utils import BoW_Estimator
from classifier.textfabric_utils import get_verses
import numpy as np
import classifier.book_data as book_data


def csvify_etcbc(
        X: list[str],
        probas: list[float],
        y_correct: list[int],
        ) -> str:
    """Convert the ETCBC verse data into a CSV-formatted string."""
    # Set header line
    result = ('"Reference","Probability for OT","Probability for NT","Correct Label",'
              + '"No. Words","ܐܠܦܒܝܬ ܣܘܪܝܝܐ","ETCBC Transliteration"\n')
    # Extract & format verse data
    for i in range(len(X)):
        result += (f'"{X[i][0]}",{probas[i][0]:.04f},{probas[i][1]:.04f},'
                   + f'{y_correct[i]}, {len(X[i][2])},"{' '.join(X[i][2])}",'
                   + f'"{' '.join(X[i][1])}"\n')
    return result


def remove_proper_nouns(verses: list) -> list:
    """Create a copy of verses with pre-defined proper nouns removed."""
    FREQUENT_PROPER_NOUN = {"JCW<", "MWC>", "J<QWB", ">JSRJL", "JWSP", ">BRHM", "CM<WN", "LJCW"}
    result_verses = []
    for i in range(len(verses)):
        # if the intersection of verse's words and FREQUENT_PROPER_NOUN exists
        if len(set(verses[i][1]) & FREQUENT_PROPER_NOUN) <= 0:
            result_verses.append(verses[i])
        else:
            # print(f"{verses[i][0]} contains a propn!")
            excludes = []
            verses_range = range(len(verses[i][1]))

            for j in verses_range:
                if verses[i][1][j] in FREQUENT_PROPER_NOUN:
                    excludes.append(j)
            
            translit_r = []
            syriac_r = []
            removed = []
            
            for j in verses_range:
                if j not in excludes:
                    translit_r.append(verses[i][1][j])
                    syriac_r.append(verses[i][2][j])
                else:
                    removed.append(verses[i][1][j])
            # print(f"Removed {removed} from verses[{i}] ({verses[i][0]})")
            result_verses.append((verses[i][0], translit_r, syriac_r))
    return result_verses


if __name__ == "__main__":
    # Parse the dataset and get verses
    # each verse in a tuple "(`verse reference`: str, `transliteration as a list of words`: list[str])"

    # extract verses from the ETCBC dataset
    ot_train_book_dict = get_verses(book_data.ot_train_books)
    ot_test_book_dict = get_verses(book_data.ot_test_books)

    nt_train_book_dict = get_verses(book_data.nt_train_books, target_fabric="etcbc/syrnt", ver="0.1")
    nt_test_book_dict = get_verses(book_data.nt_test_books, target_fabric="etcbc/syrnt", ver="0.1")
    print(nt_train_book_dict["Matthew"][0])

    # merge dictionaries for training
    train_data_dict = ot_train_book_dict.copy()
    train_data_dict.update(nt_train_book_dict)

    # transform per-book dictionaries into a list of verses
    ot_train_verses = [vrs for verses in ot_train_book_dict.values() for vrs in verses]
    nt_train_verses = [vrs for verses in nt_train_book_dict.values() for vrs in verses]
    ot_test_verses = [vrs for verses in ot_test_book_dict.values() for vrs in verses]
    nt_test_verses = [vrs for verses in nt_test_book_dict.values() for vrs in verses]

    # merge verse lists for training
    train_verse_txts = ot_train_verses.copy()
    train_verse_txts.extend(nt_train_verses)
    print(train_verse_txts[0])

    # prepare labels for training
    train_verse_labels = [0 for i in range(len(ot_train_verses))]
    train_verse_labels.extend([1 for i in range(len(nt_train_verses))])
    train_verse_labels = np.array(train_verse_labels)

    # prepare labels for evaluation
    ot_test_verse_labels = [0 for i in range(len(ot_test_verses))]
    nt_test_verse_labels = [1 for i in range(len(nt_test_verses))]

    # train classifier
    est = BoW_Estimator(MultinomialNB(), ' '.join)
    est.fit(train_verse_txts, train_verse_labels)

    # evaluate and save results
    (ot_probas, nt_probas,
     ot_mislabels, nt_mislabels) = evaluate_classifier(
                                                est,
                                                ot_test_verses, ot_test_verse_labels,
                                                nt_test_verses, nt_test_verse_labels,
                                           )
    save_mislabels(ot_mislabels, nt_mislabels,
                   formatter=csvify_etcbc,
                   ot_save_file="./out/etcbc_mnb_prediction_mislabels_ot.csv",
                   nt_save_file="./out/etcbc_mnb_prediction_mislabels_nt.csv"
                   )
    save_all_preds(ot_test_verses, ot_probas, ot_test_verse_labels,
                   nt_test_verses, nt_probas, nt_test_verse_labels,
                   formatter=csvify_etcbc,
                   ot_save_file="./out/etcbc_mnb_prediction_all_ot.csv",
                   nt_save_file="./out/etcbc_mnb_prediction_all_nt.csv"
                   )

    # Remove a few common proper nouns
    print("\nRemove common proper nouns from training verses")
    train_verses_removed = remove_proper_nouns(train_verse_txts)
    # assert(train_verses_removed != train_verse_txts)

    mnb_r = BoW_Estimator(MultinomialNB(), ' '.join)
    mnb_r.fit(train_verses_removed, train_verse_labels)

    # evaluate and save results
    (ot_probas_r, nt_probas_r,
     ot_mislabels_r, nt_mislabels_r) = evaluate_classifier(
                                                mnb_r,
                                                ot_test_verses, ot_test_verse_labels,
                                                nt_test_verses, nt_test_verse_labels,
                                           )
    save_mislabels(ot_mislabels_r, nt_mislabels_r,
                   formatter=csvify_etcbc,
                   ot_save_file="./out/etcbc_mnb_prediction_mislabels_ot_removed.csv",
                   nt_save_file="./out/etcbc_mnb_prediction_mislabels_nt_removed.csv"
                   )
    save_all_preds(ot_test_verses, ot_probas_r, ot_test_verse_labels,
                   nt_test_verses, nt_probas_r, nt_test_verse_labels,
                   formatter=csvify_etcbc,
                   ot_save_file="./out/etcbc_mnb_prediction_all_ot_removed.csv",
                   nt_save_file="./out/etcbc_mnb_prediction_all_nt_removed.csv"
                   )

    print("\nRemove common proper nouns from both training & test verses")
    train_verses_removed = remove_proper_nouns(train_verse_txts)
    ot_test_verses_r = remove_proper_nouns(ot_test_verses)
    nt_test_verses_r = remove_proper_nouns(nt_test_verses)
    # assert(train_verses_removed != train_verse_txts)

    mnb_r = BoW_Estimator(MultinomialNB(), ' '.join)
    mnb_r.fit(train_verses_removed, train_verse_labels)

    # evaluate and save results
    (ot_probas_r, nt_probas_r,
     ot_mislabels_r, nt_mislabels_r) = evaluate_classifier(
                                                mnb_r,
                                                ot_test_verses_r, ot_test_verse_labels,
                                                nt_test_verses_r, nt_test_verse_labels,
                                           )
    save_mislabels(ot_mislabels_r, nt_mislabels_r,
                   formatter=csvify_etcbc,
                   ot_save_file="./out/etcbc_mnb_prediction_mislabels_ot_removed_both.csv",
                   nt_save_file="./out/etcbc_mnb_prediction_mislabels_nt_removed_both.csv"
                   )
    save_all_preds(ot_test_verses, ot_probas_r, ot_test_verse_labels,
                   nt_test_verses, nt_probas_r, nt_test_verse_labels,
                   formatter=csvify_etcbc,
                   ot_save_file="./out/etcbc_mnb_prediction_all_ot_removed_both.csv",
                   nt_save_file="./out/etcbc_mnb_prediction_all_nt_removed_both.csv"
                   )

