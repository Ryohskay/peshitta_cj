from classifier.load_cal import load_cal_dataset

if __name__ == "__main__":
    loaded = load_cal_dataset("./")
    loaded.save_as_json("./hf_dataset/")
