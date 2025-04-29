from src.classifier.load_cal import load_cal_dataset
from src.classifier.textfabric_utils import load_etcbc_dataset

if __name__ == "__main__":
    loaded_etc = load_etcbc_dataset()
    loaded_etc.save_as_json("./hf_dataset/etcbc/")
