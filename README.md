# Peshitta CJ

[GitHub](https://github.com/Ryohskay/peshitta_cj)

## Purpose

Categorise verses from OT Peshitta (a *Classical Syriac* Translation of the Old Testament) with unknown authorship, into books translated by Jews and by Christians, through training ML classifiers to assign probabilities.

## How to run the code

***NOTE: This project is based on scikit-learn, which only supports CPU-based training.***
Because of this, this project is very slow and can easily take a day to finish a whole script
if you pack it with many classifier configurations

### Dependencies

In order to make sure that the software runs correctly, it is recommended to install the [poetry](https://python-poetry.org/) project manager.

After installing poetry, simply run: `poetry install`.

## Executing the code

To run any program in this project, execute:

- `poetry run python -m src.scripts.ensure_dirs.py`
- `poetry run python -m src.SUB_PACKAGE.FILE_NAME_STEM`

On Unix, you can customise the `run_classifiers.sh` script to run the code you want.

## Software Maintenance Manual

When developing the code, use the poetry project manager as described in the User Manual.

The code has been developed using Visual Studio Code, and it is the recommended editor for this project.

To build the documentation using an auto-loading development server, execute \verb|make build| in the docs/ directory.

To run the tests, simply execute `poertry run pytest`.

## Further development

- For developing a UI, you can use the incomplete code in `src/ui_web` subpackage as a base.
  - The experimental script using [LIME](https://github.com/marcotcr/lime) may be helpful as well.

- For implementing neural classifiers, look at the code under `neural` directory as well as the notebook `notebooks/bert-aa.ipynb` and `notebooks/bert-fine-tuning.ipynb`.

## Repository Structure

- `assets`: Assets to be used in unit tests.
- `docs/`: Sphinx directory
- `src/`: python scripts and utility files.
  - `shared/`: utilities and constants to be shared among multiple modules
  - `scraper/`: code to extract Peshitta texts from the [CAL (Comprehensive Aramaic Lexicon, Stephen A. Kaufman et al.)](https://cal.huc.edu/)
  - `classifier/`: code to build and evaluate classifiers for categorising Peshitta verses.
- `notebooks/`: Jupyter notebook files for experiments. Note that some of the code may not work as these were only used as test bed before starting full implementations.
- `test/`: `pytest` unit tests.

## System Design

- The code and system will be based on the Latin Alphabet transliteration (*romanisation*) used on the CAL.
- This implementation ignores vocalisation in NT Peshitta, since it's unnecessary for the purpose of this project.
  - Vocalisation is not indicated on the CAL OT Peshitta.
  - Vocalisation of OT Peshitta is likely to include works done much later in the history, which may negatively impact the objective reliability of the categorisation by this system.

## Author

This project is a work by Ryosuke Nagata submitted as his dissertation project for Computing, MA., Hons. at the University of Aberdeen.
