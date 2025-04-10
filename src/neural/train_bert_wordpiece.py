# Copyright 2023 Moi, A., & Patry, N.
# HuggingFace's Tokenizers.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a wordpiece tokenizer for BERT models.

This program is distributed under the terms of the Apache License version 2.0.
See ``apache-license`` file in this directory for the license.

This program was adapted from `HuggingFace's tokenizer library examples
<https://github.com/huggingface/tokenizers/blob/main/bindings/python/examples/train_bert_wordpiece.py>`.
"""

import argparse
import datasets

from tokenizers import BertWordPieceTokenizer


# Code adapted from https://github.com/huggingface/tokenizers/blob/main/bindings/python/examples/train_with_datasets.py
# Accessed: 10 May 2025
def batch_iterator():
    batch_size = 1000
    for batch in dataset.iter(batch_size=batch_size):
        yield batch["text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        default=None,
        metavar="path",
        type=str,
        required=True,
        help="The files to use as training; accept '**/*.txt' type of patterns \
                              if enclosed in quotes",
    )
    parser.add_argument(
        "--out",
        default="./",
        type=str,
        help="Path to the output directory, where the files will be saved",
    )
    parser.add_argument("--name", default="syrbert-wordpiece", type=str,
                        help="The name of the output vocab files")
    args = parser.parse_args()

    # Initialize an empty tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=True,
        lowercase=True,
    )

    # prepare a training dataset
    # Build an iterator over this dataset
    dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    # And then train
    tokenizer.train_from_iterator(
        vocab_size=10000,
        min_frequency=2,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        limit_alphabet=1000,
        wordpieces_prefix="##",
    )

    # Save the files
    tokenizer.save_model(args.out, args.name)
