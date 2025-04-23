from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={
        "train": [
            "src/classifier/hf_dataset/etcbc_train_data_0.json",
            "src/classifier/hf_dataset/etcbc_train_data_1.json",
        ],
        "validation": [
            "src/classifier/hf_dataset/etcbc_test_data_0.json",
            "src/classifier/hf_dataset/etcbc_test_data_1.json",
        ],
    },
)

print(dataset["train"][0])
