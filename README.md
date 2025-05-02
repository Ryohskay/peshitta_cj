# Peshitta CJ

[GitHub](https://github.com/Ryohskay/peshitta_cj)

## Purpose

Categorise verses from OT Peshitta (a *Classical Syriac* Translation of the Old Testament) with unknown authorship, into books translated by Jews and by Christians, through training ML classifiers to assign probabilities.

## User Manual

### Dependencies
In order to make sure that the software runs correctly, it is recommended to install the [poetry](https://python-poetry.org/) project manager.

## Executing the code
To run any program in this project, execute:

`poetry run python -m src.SUB_PACKAGE.FILE_NAME_STEM`

## Software Maintenance Manual

When developing the code, use the poetry project manager as described in the User Manual.

The code has been developed using Visual Studio Code, and it is the recommended editor for this project.

To build the documentation using an auto-loading development server, execute \verb|make build| in the docs/ directory.

To run the tests, simply execute `poertry run pytest`.

## Repository Structure

- `docs/`: Sphinx directory
- `src/`: python scripts and utility files 
  - `shared/`: utilities and constants to be shared among multiple modules
  - `scraper/`: code to extract Peshitta texts from the [CAL (Comprehensive Aramaic Lexicon, Stephen A. Kaufman et al.)](https://cal.huc.edu/)
  - `classifier/`: code to build and evaluate classifiers for categorising Peshitta verses.
- `notebooks/`: Jupyter notebook files for experiments. Note that some of the code may not work as these were only used as test bed before starting full implementations.

## System Design

- The code and system will be based on the Latin Alphabet transliteration (*romanisation*) used on the CAL.
- This implementation ignores vocalisation in NT Peshitta, since it's unnecessary for the purpose of this project.
  - Vocalisation is not indicated on the CAL OT Peshitta.
  - Vocalisation of OT Peshitta is likely to include works done much later in the history, which may negatively impact the objective reliability of the categorisation by this system.

## Author

This project is a work by Ryosuke Nagata submitted as his dissertation project for Computing, MA., Hons. at the University of Aberdeen.
