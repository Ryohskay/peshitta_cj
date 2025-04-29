# Code adapted from:
# https://huggingface.co/docs/transformers/en/training (Acc. 22 March 2025)
from datasets import load_dataset
from huggingface_hub import notebook_login
from transformers import (
    AutoTokenizer,
)

dataset = load_dataset(
    "json",
    data_files={
        "train": [
            "src/hf_dataset/etcbc_train_data_0.json",
            "src/hf_dataset/etcbc_train_data_1.json",
        ],
        "validation": [
            "src/hf_dataset/etcbc_test_data_0.json",
            "src/hf_dataset/etcbc_test_data_1.json",
        ],
    },
)

print(dataset["train"][0])

# MODEL_ID = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
# MODEL_ID = "google-bert/bert-base-multilingual-cased"
# MODEL_ID = "facebook/nllb-200-distilled-600M"
# MODEL_ID = "google/gemma-3-4b-it"
MODEL_ID = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

res = tokenizer("ܐܠܦܒܝܬ ܣܘܪܝܝܐ")
print(res)
print(tokenizer.decode(res["input_ids"]))

def tokenize_function(data):
    return tokenizer(data["text"], padding="max_length", truncation=True)

tokenized_data = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
)

print(tokenized_data)