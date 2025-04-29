from src.classifier.load_cal import load_cal_dataset
from src.classifier.textfabric_utils import load_etcbc_dataset

if __name__ == "__main__":
    loaded_etc = load_etcbc_dataset()
    print(len(loaded_etc.test.verses[0]))
    loaded_etc.save_as_json("./src/hf_dataset/etcbc/", mode="translit")
    loaded_cal = load_cal_dataset("./src")
    print(len(loaded_cal.test.verses[0]))
    loaded_cal.save_as_json("./src/hf_dataset/cal/", mode="translit")
