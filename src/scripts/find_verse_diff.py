from src.classifier.textfabric_utils import load_etcbc_dataset
from src.classifier.load_cal import load_cal_dataset


def compare_verse_references(etcbc_dataset, cal_dataset):
    """Compare verse references between ETCBC and CAL datasets.

    Args:
        etcbc_dataset: Loaded ETCBC dataset (LoadedDataset instance).
        cal_dataset: Loaded CAL dataset (LoadedDataset instance).

    Returns:
        A tuple containing:
            - verses_in_etcbc_not_in_cal: List of verse references in ETCBC but not in CAL.
            - verses_in_cal_not_in_etcbc: List of verse references in CAL but not in ETCBC.
    """
    # Extract verse references from ETCBC dataset
    etcbc_verse_refs = [verse.reference for verse in etcbc_dataset.test.get_samples()]

    # Extract verse references from CAL dataset
    cal_verse_refs = [verse.reference for verse in cal_dataset.test.get_samples()]

    # Find differences
    etcbc_verses = []
    cal_verses = []
    for i in range(len(etcbc_verse_refs)):
        etcbc_verses.append(etcbc_verse_refs[i])
        cal_verses.append(cal_verse_refs[i])
        print(f"ETCBC: {etcbc_verse_refs[i]}\t"+ f"CAL: {cal_verse_refs[i]}")

    return None


if __name__ == "__main__":
    # Load datasets
    print("Loading ETCBC dataset...")
    etcbc_dataset = load_etcbc_dataset()

    print("Loading CAL dataset...")
    cal_dataset = load_cal_dataset("./src")

    # # Compare verse references
    # print("Comparing verse references...")
    # verses_in_etcbc_not_in_cal, verses_in_cal_not_in_etcbc = compare_verse_references(
    #     etcbc_dataset, cal_dataset
    # )
    compare_verse_references(etcbc_dataset, cal_dataset)

    # # Print results
    # print("\nVerses in ETCBC but not in CAL:")
    # for verse in sorted(verses_in_etcbc_not_in_cal):
    #     print(verse)

    # print("\nVerses in CAL but not in ETCBC:")
    # for verse in sorted(verses_in_cal_not_in_etcbc):
    #     print(verse)