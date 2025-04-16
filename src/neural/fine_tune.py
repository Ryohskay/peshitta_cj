import numpy as np
from datasets import Dataset, load_dataset
from torch.nn import LayerNorm, functional
from torch.optim import AdamW, lr_scheduler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_pt_utils import get_parameter_names

dataset = load_dataset("json",
                       data_files={"train": ["src/classifier/hf_dataset/etcbc_train_data_0.json", "src/classifier/hf_dataset/etcbc_train_data_1.json"],
                                   "validation": ["src/classifier/hf_dataset/etcbc_test_data_0.json", "src/classifier/hf_dataset/etcbc_test_data_1.json"]},
                                   )

print(dataset["train"][0])