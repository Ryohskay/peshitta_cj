# What to do now

## ETCBC dataset

* Output a datasheet of char n-grams occuring in both OT and NT.
* Random Forest
* BERT (ensemble learning?) Follow (Yamshchikov et al., 2022) and (Alqurashi et al., 2025)
 * Sentencepiece tokenizer to implement sub-word tokenization in Classical Syriac
  * follow [mt-empty/assyrian-translation-model] on GitHub
  * special tokens? (bos/eos tokens, etc.)
  * padding?
  * vocabulary size? (probably need downstream task performance check to fine-tune it)
   * 32k [recommended for NMT by Kudo (the guy invented unigram model)](https://github.com/google/sentencepiece/issues/415#issuecomment-550165568), he also suggests 8k / 16k for "smaller" corpora.
 * Perform transfer learning on:
  * CAMeLBERT for Classical Arabic (CAMeL-Lab/bert-base-arabic-camelbert-ca)
  * BEREL for Rabbinic Hebrew (dicta-il/BEREL_2.0)
 * Ensemble Learning?
  * Per-author models voting (Alqurashi et al., 2025)
  * Logistic Regression on the outputs of BERT & other logistic regression classifiers (Fabien et al., 2020)
 * Tuning hyperparameters?
  * (L2) Regularisation: AdamW?

## CAL dataset

* MultinomialNB (Word & character n-gram, with & without proper nouns (PN, GN))
* SVC (Character n-gram)
* Random Forest (Character n-gram)
* BERT

## Intrepretation

* Besides linguistic features pointed out by Jakub:
 * LIME (Ribeiro et al., 2016)
